import gc
import platform
import os
import subprocess as sp
import time

import numpy as np
import gradio as gr
import json
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception:
    pass
import torchaudio
import torchvision
from decord import VideoReader
from decord import cpu
import math
import einops
import torchvision.transforms as transforms

from einops import rearrange
from torch.nn import functional as F
from torchaudio import transforms as T

from transformers import Qwen2_5OmniProcessor

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

_AUDIO_INPUT_SR = 44100
_AUDIO_INPUT_SECONDS = 10

_SEGMENT_DURATION = 11
_OVERLAP_DURATION = 1
_MAX_SHORT_DURATION = 10

_UNDERSTANDING_MAX_HISTORY_TOKENS = 2048


def create_spectrogram_image(audio, sample_rate=44100, figsize=(12, 4), colormap='magma',
                             min_duration=10.0):
    """
    Create spectrogram using matplotlib.
    audio: torch.Tensor or numpy.ndarray, shape (channels, samples) or (samples,)
    sample_rate: Sample rate
    figsize: Figure size (width, height)
    colormap: Colormap name, default 'magma' (purple-red tone)
    min_duration: Minimum x-axis duration in seconds (pads display if audio is shorter)
    Returns: PIL Image
    """
    try:
        import librosa
        use_librosa = True
    except ImportError:
        use_librosa = False

    hop_length = 512

    # Convert to numpy array and ensure on CPU
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.array(audio)

    # If multi-channel, take first channel or average
    if audio_np.ndim > 1:
        if audio_np.shape[0] <= 2:  # (channels, samples)
            audio_np = audio_np[0] if audio_np.shape[0] == 1 else np.mean(audio_np, axis=0)
        else:  # (samples, channels)
            audio_np = np.mean(audio_np, axis=1)

    # Ensure float32 format
    audio_np = audio_np.astype(np.float32)

    # If int16 format, convert to float32 and normalize to [-1, 1]
    if audio_np.dtype == np.int16 or np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np.astype(np.float32) / 32767.0
    else:
        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            audio_np = audio_np / max_val

    actual_duration = len(audio_np) / sample_rate
    display_duration = max(actual_duration, min_duration)

    # Compute mel spectrogram
    if use_librosa:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=128,
            win_length=2048,
            center=True,
            power=1.0
        )
    else:
        from scipy import signal
        f, t, Zxx = signal.stft(audio_np, fs=sample_rate, nperseg=2048, noverlap=2048-hop_length)
        mel_spec = np.abs(Zxx)

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec ** 2, ref=np.max) if use_librosa else 20 * np.log10(np.maximum(mel_spec, 1e-10))

    # Time axis: each column = hop_length / sample_rate seconds
    n_frames = mel_spec_db.shape[1]
    time_extent = n_frames * hop_length / sample_rate  # actual time span of the spectrogram

    # Create figure — white outer background, black only inside the axes (pad region)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('black')
    im = ax.imshow(
        mel_spec_db,
        aspect='auto',
        origin='lower',
        cmap=colormap,
        interpolation='nearest',
        vmin=mel_spec_db.max() - 80,
        vmax=mel_spec_db.max(),
        extent=[0, time_extent, 0, mel_spec_db.shape[0]],
    )
    ax.set_xlim(0, display_duration)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency', fontsize=9)
    ax.set_title(f'Mel Spectrogram ({actual_duration:.2f}s)', fontsize=10)
    ax.tick_params(colors='black')
    plt.colorbar(im, ax=ax, label='dB', pad=0.02)
    plt.tight_layout(pad=0.5)

    # Convert to PIL Image — keep full resolution
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    return img


def crossfade_audio(audio1: torch.Tensor, audio2: torch.Tensor, fade_samples: int) -> torch.Tensor:
    """
    Crossfade two audio segments.
    audio1: First audio segment
    audio2: Second audio segment
    fade_samples: Number of samples for crossfade
    Returns: Concatenated audio
    """
    fade_in = torch.linspace(0, 1, fade_samples, device=audio1.device)
    fade_out = torch.linspace(1, 0, fade_samples, device=audio1.device)
    
    if audio1.dim() == 1:
        audio1_fade = audio1[-fade_samples:] * fade_out
        audio2_fade = audio2[:fade_samples] * fade_in
        crossfaded = audio1_fade + audio2_fade
        result = torch.cat([audio1[:-fade_samples], crossfaded, audio2[fade_samples:]])
    else:
        # 2D audio [channels, samples]
        fade_in = fade_in.unsqueeze(0)
        fade_out = fade_out.unsqueeze(0)
        audio1_fade = audio1[:, -fade_samples:] * fade_out
        audio2_fade = audio2[:, :fade_samples] * fade_in
        crossfaded = audio1_fade + audio2_fade
        result = torch.cat([audio1[:, :-fade_samples], crossfaded, audio2[:, fade_samples:]], dim=1)
    
    return result

# Global variables for model configuration storage
model_configurations = {}
device = torch.device("cpu")  # Default device

# Set temp directory
os.environ['TMPDIR'] = '/aifs4su/data/tianzeyue/tmp'

current_model_name = None
current_model = None
current_sample_rate = None
current_sample_size = None


_SYNC_SIZE = 224
from torchvision.transforms import v2        
sync_transform = v2.Compose([
    v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


from mmaudio.model.utils.features_utils import FeaturesUtils

_SYNCHFORMER_CKPT = os.environ.get(
    "SYNCHFORMER_CKPT",
    "/mnt/shanghai2cephs/zeyuetian/project/MMAudio/ext_weights/synchformer_state_dict.pth",
)
_QWEN_OMNI_MODEL_PATH = os.environ.get(
    "QWEN_OMNI_MODEL_PATH",
    "/mnt/shangcephfs/mm-base-vision-ascend/zeyuetian/AudioX-private/model/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e",
)

sync_feature_extractor = None
qwen_processor = None

def _get_sync_feature_extractor():
    global sync_feature_extractor
    if sync_feature_extractor is None:
        sync_feature_extractor = FeaturesUtils(
            tod_vae_ckpt='vae_path',
            enable_conditions=True,
            bigvgan_vocoder_ckpt='bigvgan_path',
            synchformer_ckpt=_SYNCHFORMER_CKPT,
            mode='mode',
        ).eval().cuda()
    return sync_feature_extractor

def _get_qwen_processor():
    global qwen_processor
    if qwen_processor is None:
        qwen_processor = Qwen2_5OmniProcessor.from_pretrained(_QWEN_OMNI_MODEL_PATH)
    return qwen_processor

def _get_omni_conditioner(model):
    """
    Extract the QwenOmni conditioner (and its .model / .processor) from a loaded
    ConditionedDiffusionModelWrapper.  Returns (conditioner, processor, thinker_model)
    or (None, None, None) if not found.
    """
    if model is None:
        return None, None, None
    conditioner_wrapper = getattr(model, "conditioner", None)
    if conditioner_wrapper is None:
        return None, None, None
    conditioners = getattr(conditioner_wrapper, "conditioners", {})
    for _key, cond in conditioners.items():
        cls_name = type(cond).__name__
        if "QwenOmni" in cls_name or "MetaQueryWithQwenOmni" in cls_name:
            return cond, getattr(cond, "processor", None), getattr(cond, "model", None)
    return None, None, None


def _offload_non_omni_to_cpu(model):
    """
    Move every part of the ConditionedDiffusionModelWrapper to CPU *except* the
    QwenOmni conditioner (which owns the thinker we need for understanding).
    This frees GPU memory while understanding is running.
    """
    if model is None:
        return
    # Move the DiT / diffusion core
    diff_model = getattr(model, "model", None)
    if diff_model is not None:
        try:
            diff_model.cpu()
        except Exception:
            pass
    # Move pretransform (VAE)
    pretransform = getattr(model, "pretransform", None)
    if pretransform is not None:
        try:
            pretransform.cpu()
        except Exception:
            pass
    # Move every conditioner except the QwenOmni one
    conditioner_wrapper = getattr(model, "conditioner", None)
    if conditioner_wrapper is not None:
        for _key, cond in getattr(conditioner_wrapper, "conditioners", {}).items():
            cls_name = type(cond).__name__
            if "QwenOmni" not in cls_name and "MetaQueryWithQwenOmni" not in cls_name:
                try:
                    cond.cpu()
                except Exception:
                    pass
    torch.cuda.empty_cache()
    gc.collect()
    print("[Understanding] Non-omni model components offloaded to CPU.")


def _restore_model_to_cuda(model):
    """
    Move the entire ConditionedDiffusionModelWrapper back to CUDA so that
    generation / editing can run normally.
    """
    if model is None:
        return
    if not torch.cuda.is_available():
        return
    try:
        model.cuda()
    except Exception as e:
        print(f"[Warning] Failed to restore model to CUDA: {e}")
    # Ensure pretransform stays float32
    pretransform = getattr(model, "pretransform", None)
    if pretransform is not None:
        try:
            pretransform.to(dtype=torch.float32)
        except Exception:
            pass
    torch.cuda.empty_cache()
    gc.collect()
    print("[Understanding] Model restored to CUDA.")


def _clip_audio_file(src_path: str, start: float, duration: float) -> str:
    """
    Trim an audio file to [start, start+duration] seconds and save to a temp file.
    Returns the temp file path, or src_path if trimming is not needed / fails.
    """
    if start <= 0 and duration <= 0:
        return src_path
    try:
        wav, sr = torchaudio.load(src_path)
        s = int(start * sr)
        e = int((start + duration) * sr) if duration > 0 else wav.shape[-1]
        wav = wav[:, s:e]
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav, sr)
        tmp.close()
        return tmp.name
    except Exception as ex:
        print(f"[warn] audio clip failed: {ex}")
        return src_path


def _clip_video_file(src_path: str, start: float, duration: float) -> str:
    """
    Trim a video file to [start, start+duration] seconds using ffmpeg and save to a temp file.
    Returns the temp file path, or src_path if trimming is not needed / fails.
    """
    if start <= 0 and duration <= 0:
        return src_path
    try:
        import tempfile, subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        cmd = ["ffmpeg", "-y", "-ss", str(start)]
        if duration > 0:
            cmd += ["-t", str(duration)]
        cmd += ["-i", src_path, "-c", "copy", tmp.name]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp.name
    except Exception as ex:
        print(f"[warn] video clip failed: {ex}")
        return src_path


def generate_understanding(
    model_name,
    text_input,
    audio_input_file,
    audio_input_path_text,
    video_input_file,
    video_input_path_text,
    av_start_time,
    av_duration,
    system_prompt,
    use_audio_in_video,
    chat_history=None,
):
    """
    Understanding entry with multi-turn context (text-only history).
    Each turn can freely combine text / audio / video.
    Previous turns are kept as text-only messages to avoid OOM.
    Returns (chatbot_messages, updated_chat_history, cleared_text_input).
    """
    global current_model_name, current_model, current_sample_rate, current_sample_size

    if chat_history is None:
        chat_history = []

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False
    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_name not in model_configurations:
        err = f"Error: model '{model_name}' configuration not found."
        chat_history.append({"role": "user", "content": text_input or "(empty)"})
        chat_history.append({"role": "assistant", "content": err})
        return chat_history, chat_history, ""

    cfg = model_configurations[model_name]
    mc_path = cfg.get("model_config")
    if mc_path:
        with open(mc_path) as f:
            model_config = json.load(f)
    else:
        model_config = None
    if current_model is None or model_name != current_model_name:
        current_model, model_config, sample_rate, sample_size = load_model(
            model_name=model_name, model_config=model_config,
            model_ckpt_path=cfg.get("ckpt_path"),
            pretrained_name=cfg.get("pretrained_name"),
            pretransform_ckpt_path=cfg.get("pretransform_ckpt_path"),
            device=device)
        current_model_name = model_name
        current_sample_rate = sample_rate
        current_sample_size = sample_size

    omni_cond, processor, omni_model = _get_omni_conditioner(current_model)
    if omni_cond is None or omni_model is None or processor is None:
        err = "Error: the loaded model does not contain a QwenOmni conditioner."
        chat_history.append({"role": "user", "content": text_input or "(empty)"})
        chat_history.append({"role": "assistant", "content": err})
        return chat_history, chat_history, ""

    _offload_non_omni_to_cpu(current_model)

    try:
        start_t = float(av_start_time) if av_start_time else 0.0
        dur_t   = float(av_duration)   if av_duration   else 0.0

        sys_text = (system_prompt or "").strip()
        if not sys_text:
            sys_text = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                "capable of perceiving auditory and visual inputs, as well as generating text and speech."
            )

        user_content = []

        video_path = None
        if video_input_file is not None:
            if isinstance(video_input_file, str) and video_input_file.strip():
                video_path = video_input_file.strip()
            elif hasattr(video_input_file, "name"):
                video_path = video_input_file.name
        if not video_path and video_input_path_text and video_input_path_text.strip():
            video_path = video_input_path_text.strip()

        video_tensor = None
        if video_path:
            video_path = _clip_video_file(video_path, start_t, dur_t)
            try:
                from decord import VideoReader, cpu as decord_cpu
                _vr = VideoReader(video_path, ctx=decord_cpu(0))
                _video_duration = len(_vr) / max(_vr.get_avg_fps(), 1e-3)
                video_tensor = video_read_local(video_path, seek_time=0., duration=_video_duration, target_fps=2)
                print(f"[Understanding] Video loaded via video_read_local: {video_tensor.shape}")
            except Exception as _e:
                print(f"[warn] video_read_local failed: {_e}, will use path fallback")
                video_tensor = None
            user_content.append({"type": "video", "video": video_path})

        audio_path = None
        if audio_input_file is not None:
            if isinstance(audio_input_file, str) and audio_input_file.strip():
                audio_path = audio_input_file.strip()
            elif hasattr(audio_input_file, "name"):
                audio_path = audio_input_file.name
        if not audio_path and audio_input_path_text and audio_input_path_text.strip():
            audio_path = audio_input_path_text.strip()
        if audio_path:
            audio_path = _clip_audio_file(audio_path, start_t, dur_t)
            user_content.append({"type": "audio", "audio": audio_path})

        if text_input and text_input.strip():
            user_content.append({"type": "text", "text": text_input.strip()})

        if not user_content:
            err = "Error: please provide at least one input (text / audio / video)."
            chat_history.append({"role": "user", "content": "(empty)"})
            chat_history.append({"role": "assistant", "content": err})
            return chat_history, chat_history, ""

        # Build display text for chatbot
        display_parts = []
        if video_path:
            display_parts.append(f"[Video: {os.path.basename(video_path)}]")
        if audio_path:
            display_parts.append(f"[Audio: {os.path.basename(audio_path)}]")
        if text_input and text_input.strip():
            display_parts.append(text_input.strip())
        user_display = " ".join(display_parts)

        # Sliding window: keep only recent history that fits within token budget.
        # Estimate ~1 token per 3 chars; trim from the front (oldest turns first),
        # always dropping user+assistant pairs together to keep roles consistent.
        trimmed_history = list(chat_history)
        total_chars = sum(len(m["content"]) for m in trimmed_history)
        max_chars = _UNDERSTANDING_MAX_HISTORY_TOKENS * 3
        while total_chars > max_chars and len(trimmed_history) >= 2:
            total_chars -= len(trimmed_history[0]["content"])
            total_chars -= len(trimmed_history[1]["content"])
            trimmed_history = trimmed_history[2:]

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        ]
        for prev in trimmed_history:
            conversation.append({
                "role": prev["role"],
                "content": [{"type": "text", "text": prev["content"]}],
            })
        conversation.append({"role": "user", "content": user_content})

        USE_AUDIO_IN_VIDEO = bool(use_audio_in_video)

        text_template = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        from qwen_omni_utils import process_mm_info
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        if video_tensor is not None:
            videos = [video_tensor.cpu()]

        inputs = processor(
            text=text_template,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        thinker = omni_model.thinker if hasattr(omni_model, "thinker") else omni_model
        thinker_device = next(thinker.parameters()).device
        inputs = {k: v.to(thinker_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs.get("input_ids")
        input_len = input_ids.shape[1] if input_ids is not None else 0

        with torch.no_grad():
            text_ids = thinker.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=512)

        generated_ids = text_ids[:, input_len:]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = output_text[0].strip() if output_text else ""
        print(f"[Understanding] Output: {result}")

        # Update chat history (text-only for context)
        user_text = text_input.strip() if text_input and text_input.strip() else user_display
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": result})

    except Exception as e:
        result = f"Error: {e}"
        chat_history.append({"role": "user", "content": text_input or "(empty)"})
        chat_history.append({"role": "assistant", "content": result})

    finally:
        _restore_model_to_cuda(current_model)

    return chat_history, chat_history, ""


def build_qwen_prompt_t2a(prompt: str) -> str:
    """
    Text-to-Audio system prompt (general sound design), aligned with infer_text_condition-audio.py
    """
    return (
        "<|im_start|>system\n"
        "You are a professional sound designer. Your task is to synthesize a high-fidelity audio clip based on the following text description."
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )



def build_qwen_prompt_t2m(prompt: str) -> str:
    """
    Text-to-Music system prompt, aligned with infer_text_condition-music.py
    """
    return (
        "<|im_start|>system\n"
        "You are a versatile music producer and composer. Your goal is to create a high-quality, original piece of music based on the user's request. "
        "Focus on melody, harmony, rhythm, and instrumentation to capture the desired mood."
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qwen_prompt_tts(prompt: str) -> str:
    """
    Text-to-Speech system prompt, aligned with infer_speech_seed.py.
    The transcript text is NOT embedded in the qwen prompt; it is passed
    separately via the 'speech_prompt' conditioning field.
    """
    return (
        "<|im_start|>system\n"
        "Generate speech from the input text."
        "<|im_end|>\n"
        "<|im_start|>user\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_qwen_prompt_v2a(prompt: str) -> str:
    """
    Video-to-Audio system prompt for foley art
    """
    return (
        "<|im_start|>system\n"
        "You are a professional foley artist. Understand this video and create the sounds for this video."
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n"
        "<|vision_bos|><|VIDEO|><|vision_eos|>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qwen_prompt_v2m(prompt: str) -> str:
    """
    Video-to-Music system prompt for background music
    """
    return (
        "<|im_start|>system\n"
        "You are a versatile music producer. Create a high-quality background music track that fits the style, energy, and mood of this video."
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n"
        "<|vision_bos|><|VIDEO|><|vision_eos|>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qwen_prompt_editing(prompt: str) -> str:
    """
    Audio Editing system prompt, aligned with infer_editing.py
    """
    return (
        "<|im_start|>system\n"
        "You are a professional sound engineer. Your task is to precisely edit the provided audio based on the user's instructions."
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[0]
    target_duration = duration * target_fps
    if current_duration > target_duration:
        video_tensor = video_tensor[:target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[-1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(repeat_times, 1, 1, 1)), dim=0)
    return video_tensor


def video_read_local(filepath, seek_time=0., duration=-1, target_fps=2):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        # Handle image file
        resize_transform = transforms.Resize((224, 224))
        image = Image.open(filepath).convert("RGB")  # Open image and convert to RGB
        frame = transforms.ToTensor()(image).unsqueeze(0)  # Convert to tensor and add batch dim [1, C, H, W]
        # Resize the image to 224x224
        frame = resize_transform(frame)
        # Assume duration and replicate frames to match target fps
        target_frames = int(duration * target_fps)
        frame = frame.repeat(int(math.ceil(target_frames / frame.shape[0])), 1, 1, 1)[:target_frames]
        assert frame.shape[0] == target_frames, f"The shape of frame is {frame.shape}"
        return frame  # [N, C, H, W]

    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    seek_frame = int(seek_time * fps)
    if duration > 0:
        total_frames_to_read = int(target_fps * duration)
        frame_interval = int(math.ceil(fps / target_fps))
        end_frame = min(seek_frame + total_frames_to_read * frame_interval, total_frames)
        frame_ids = list(range(seek_frame, end_frame, frame_interval))
    else:
        frame_interval = int(math.ceil(fps / target_fps))
        frame_ids = list(range(0, total_frames, frame_interval))

    # Batch read specified frames
    frames = vr.get_batch(frame_ids).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

    if frames.shape[2] != 224 or frames.shape[3] != 224:
        # Resize only when necessary
        print(f'resizing...--->224x224')
        resize_transform = transforms.Resize((224, 224))
        frames = resize_transform(frames)

    # Adjust video duration
    video_tensor = adjust_video_duration(frames, duration, target_fps)

    assert video_tensor.shape[0] == duration * target_fps, f"The shape of video_tensor is {video_tensor.shape}"

    return video_tensor


def merge_video_audio(video_path, audio_path, output_path, start_time, duration):
    command = [
        'ffmpeg',
        '-y',                   # Overwrite output files without asking
        '-ss', str(start_time), # Start time
        '-t', str(duration),    # Duration
        '-i', video_path,       # Input video file
        '-i', audio_path,       # Input audio file
        '-c:v', 'copy',         # Copy the video codec (no re-encoding)
        '-c:a', 'aac',          # Use AAC audio codec
        '-map', '0:v:0',        # Map the video from the first input
        '-map', '1:a:0',        # Map the audio from the second input
        '-shortest',            # Stop encoding when the shortest input ends
        '-strict', 'experimental',  # Allow experimental codecs if needed
        output_path             # Output file path
    ]
    
    try:
        sp.run(command, check=True)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None

def load_model(model_name, model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model_configurations
    start_time = time.time()
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load model same as infer_speech_seed.py to ensure pretransform weights load correctly
        # Use load_state_dict directly (not copy_state_dict), prefer strict=True for full key match
        state_dict = load_ckpt_state_dict(model_ckpt_path)
        # Load to CPU first to avoid OOM
        # Prefer strict=True (default) to ensure all keys match including pretransform
        # On failure, fallback to strict=False but ensure pretransform keys are handled
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: strict=True loading failed: {e}")
            print("Falling back to strict=False loading...")
            # Use strict=False loading but ensure pretransform keys load correctly
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys (will use default initialization): {missing_keys[:10]}...")
            if unexpected_keys:
                print(f"Unexpected keys (will be ignored): {unexpected_keys[:10]}...")
            # Check pretransform-related keys
            pretransform_missing = [k for k in missing_keys if 'pretransform' in k]
            if pretransform_missing:
                print(f"WARNING: Missing pretransform keys: {pretransform_missing[:10]}...")
                print("This may cause precision issues in VAE decoding!")

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        pretransform_state_dict = load_ckpt_state_dict(pretransform_ckpt_path)
        # Ensure pretransform loads with strict=False for checkpoint format compatibility
        if hasattr(model, 'pretransform') and model.pretransform is not None:
            model.pretransform.load_state_dict(pretransform_state_dict, strict=False)
        print(f"Done loading pretransform")

    # Move model to device and set to eval mode
    model.to(device).eval().requires_grad_(False)

    # Ensure pretransform always uses float32 to avoid precision loss
    # Consistent with generate_diffusion_cond (see generation.py:264)
    if hasattr(model, 'pretransform') and model.pretransform is not None:
        model.pretransform = model.pretransform.to(dtype=torch.float32).eval()

    if model_half:
        # If using half precision, keep pretransform in float32
        if hasattr(model, 'pretransform') and model.pretransform is not None:
            # pretransform must stay float32, so save it first
            pretransform_backup = model.pretransform
            model.pretransform = None
            model.to(torch.float16)
            # Restore pretransform and keep float32
            model.pretransform = pretransform_backup.to(dtype=torch.float32).eval()
        else:
            model.to(torch.float16)
    
    load_time = time.time() - start_time
    print(f"Done loading model {model_name} (took {load_time:.2f}s)")

    return model, model_config, sample_rate, sample_size


def load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total):
    audio_tensor, sr = torchaudio.load(audio_path)
    # Ensure audio length is `seconds_total` seconds
    start_index = int(sample_rate * seconds_start)  # Compute start index
    target_length = int(sample_rate * seconds_total)
    end_index = start_index + target_length  # Compute end index
    audio_tensor = audio_tensor[:, start_index:end_index]  # Slice audio tensor
    if audio_tensor.shape[1] < target_length:
        pad_length = target_length - audio_tensor.shape[1]
        audio_tensor = F.pad(audio_tensor, (pad_length, 0))  # Pad at start

    if audio_tensor.shape[0] == 2:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=False)

    return audio_tensor


def _get_audio_duration(audio_path: str) -> float:
    """Return the duration (seconds) of an audio file, or 0.0 on failure."""
    if not audio_path or not os.path.exists(audio_path):
        return 0.0
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        try:
            wav, sr = torchaudio.load(audio_path)
            return wav.shape[-1] / sr
        except Exception:
            return 0.0


def _load_audio_input_wav(audio_input_path: str, sample_rate: int, seconds_total: int) -> torch.Tensor:
    """
    Read audio_input_path (file path) and process to 1D Tensor:
    - mp3: prefer pedalboard if available, else fallback to torchaudio
    - stereo -> mono
    - resample to sample_rate
    - crop/pad to fixed length sample_rate * seconds_total
    Reference: load_audio + post-processing in data/custom_metadata_qwen-feature.py
    """
    sample_size = int(sample_rate * seconds_total)
    if audio_input_path is None or str(audio_input_path).strip() == "":
        return torch.zeros(sample_size, dtype=torch.float32)

    audio_input_path = str(audio_input_path).strip()
    ext = os.path.splitext(audio_input_path)[1].lower().lstrip(".")

    wav = None
    sr = sample_rate

    if ext == "mp3":
        try:
            from pedalboard.io import AudioFile  # type: ignore
            with AudioFile(audio_input_path) as f:
                wav_np = f.read(f.frames)
                wav = torch.from_numpy(wav_np)
                sr = int(f.samplerate)
        except Exception as e:
            print(f"[warn] mp3 load failed via pedalboard for {audio_input_path}: {e}; fallback to torchaudio")

    if wav is None:
        try:
            wav, sr = torchaudio.load(audio_input_path, format=ext if ext else None)
            sr = int(sr)
        except Exception as e:
            print(f"[warn] audio_input load failed for {audio_input_path}: {e}; use zeros")
            return torch.zeros(sample_size, dtype=torch.float32)

    # wav shape: [C, T] or [T]
    if wav.dim() == 2 and wav.shape[0] == 2:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # resample to target sample_rate
    if sr != sample_rate:
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            wav = resampler(wav)
        except Exception as e:
            print(f"[warn] resample failed (sr={sr} -> {sample_rate}) for {audio_input_path}: {e}; use zeros")
            return torch.zeros(sample_size, dtype=torch.float32)

    # ensure 1D
    if wav.dim() > 1:
        wav = wav.squeeze(0)

    wav = wav.to(torch.float32)
    cur_len = int(wav.shape[0])
    if cur_len > sample_size:
        wav = wav[:sample_size]
    elif cur_len < sample_size:
        wav = F.pad(wav, (0, sample_size - cur_len), "constant", 0.0)

    return wav


def _trim_silence(audio: torch.Tensor, sample_rate: int,
                  top_db: float = 30.0, frame_length: int = 2048,
                  hop_length: int = 512, min_length_ms: int = 200) -> torch.Tensor:
    """
    Trim leading and trailing silence from int16 audio tensor [channels, samples].
    Uses energy-based detection: frames below `top_db` dB relative to peak are silence.
    Keeps at least `min_length_ms` milliseconds of audio.
    """
    import numpy as np
    if audio.numel() == 0:
        return audio
    # Work on mono for detection
    if audio.dim() == 2:
        mono = audio.float().mean(dim=0).numpy()
    else:
        mono = audio.float().numpy()

    # Compute frame energy in dB
    n = len(mono)
    if n < frame_length:
        return audio
    num_frames = 1 + (n - frame_length) // hop_length
    energy = np.array([
        np.mean(mono[i * hop_length : i * hop_length + frame_length] ** 2)
        for i in range(num_frames)
    ])
    energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
    threshold = energy_db.max() - top_db

    # Find first and last frame above threshold
    active = np.where(energy_db >= threshold)[0]
    if len(active) == 0:
        return audio

    start_sample = int(active[0] * hop_length)
    end_sample = int(min(active[-1] * hop_length + frame_length, n))

    # Ensure minimum length
    min_samples = int(sample_rate * min_length_ms / 1000)
    if end_sample - start_sample < min_samples:
        center = (start_sample + end_sample) // 2
        start_sample = max(0, center - min_samples // 2)
        end_sample = min(n, start_sample + min_samples)

    if audio.dim() == 2:
        trimmed = audio[:, start_sample:end_sample]
    else:
        trimmed = audio[start_sample:end_sample]

    print(f"[TTS] Trimmed silence: {n} -> {end_sample - start_sample} samples "
          f"({start_sample / sample_rate:.2f}s ~ {end_sample / sample_rate:.2f}s)")
    return trimmed


def _auto_asr_voice_prompt(audio_path: str, model, device) -> str:
    """
    Use the QwenOmni thinker to transcribe voice prompt audio via ASR.
    Returns the transcribed text, or empty string on failure.
    """
    try:
        omni_cond, processor, omni_model = _get_omni_conditioner(model)
        if omni_cond is None or omni_model is None or processor is None:
            print("[ASR] No QwenOmni conditioner found, skipping auto ASR")
            return ""

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition assistant. Transcribe the given audio accurately."}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Please transcribe the speech in this audio clip. Output only the transcription text, nothing else."},
            ]},
        ]

        text_template = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        from qwen_omni_utils import process_mm_info
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text_template, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )

        thinker = omni_model.thinker if hasattr(omni_model, "thinker") else omni_model
        thinker_device = next(thinker.parameters()).device
        inputs = {k: v.to(thinker_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs.get("input_ids")
        input_len = input_ids.shape[1] if input_ids is not None else 0

        with torch.no_grad():
            text_ids = thinker.generate(**inputs, use_audio_in_video=False, max_new_tokens=256)

        generated_ids = text_ids[:, input_len:]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = output_text[0].strip() if output_text else ""
        print(f"[ASR] Transcription: {result}")
        return result
    except Exception as e:
        print(f"[ASR] Auto transcription failed: {e}")
        return ""


def generate_cond(
        prompt,
        model_name,
        task_type="T2A",           # Task type T2A / T2M (aligned with inputs order)
        video_file=None,           # File upload
        video_path=None,
        sync_feature_path=None,
        audio_prompt_file=None,   # File upload
        audio_prompt_path=None,
        audio_input_path=None,    # Audio input (file path), aligned with training audio_input_prompt
        speech_prompt_file=None,   # Voice prompt audio file upload (filepath)
        speech_prompt_path=None,   # Voice prompt audio path (text input)
        voice_ref_text=None,       # Transcript of the voice prompt audio (for VC-style TTS)
        seconds_start=0,
        seconds_total=10,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1,
    ):

    global current_model_name, current_model, current_sample_rate, current_sample_size

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Restore model to CUDA in case it was offloaded during understanding
    _restore_model_to_cuda(current_model)

    print(f"Prompt: {prompt}")
    print(f"Model: {model_name}")

    preview_images = []
    if preview_every == 0:
        preview_every = None
        
    print(f'seconds_total: {seconds_total}')
    try:
        length_set = int(seconds_total)
    except ValueError:
        print("Invalid input for seconds_total, using default value of 10.")
        length_set = 10  # Default fallback value
    print(f'length_set: {length_set}')
    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_name not in model_configurations:
        raise ValueError(f"Model {model_name} configuration is not available.")

    cfg = model_configurations[model_name]
    model_config_path = cfg.get("model_config")
    ckpt_path = cfg.get("ckpt_path")
    pretrained_name = cfg.get("pretrained_name")
    pretransform_ckpt_path = cfg.get("pretransform_ckpt_path")
    model_type = cfg.get("model_type", "diffusion_cond")

    if model_config_path:
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None
    target_fps = model_config.get("video_fps", 5)

    if current_model is None or model_name != current_model_name:
        current_model, model_config, sample_rate, sample_size = load_model(
            model_name=model_name,
            model_config=model_config,
            model_ckpt_path=ckpt_path,
            pretrained_name=pretrained_name,
            pretransform_ckpt_path=pretransform_ckpt_path,
            device=device,
        )
        current_model_name = model_name
        model = current_model
        current_sample_rate = sample_rate
        current_sample_size = sample_size
    else:
        model = current_model
        sample_rate = current_sample_rate
        sample_size = current_sample_size

    # Handle video input: support both file upload and path, allow no video for pure T2A/T2M
    if video_file is not None:
        # Gradio File may return object / dict / path string
        if isinstance(video_file, dict) and "name" in video_file:
            video_path = video_file["name"]
        elif hasattr(video_file, "name"):
            video_path = video_file.name  # Temp path of uploaded file
        elif isinstance(video_file, str):
            video_path = video_file.strip()
        else:
            video_path = None
    elif video_path:  # If no upload, use input path
        video_path = video_path.strip()
    else:
        video_path = None

    # Handle audio input: support file upload and path, allow empty for pure T2A/T2M
    if audio_prompt_file is not None:
        print(f'audio_prompt_file: {audio_prompt_file}')
        if isinstance(audio_prompt_file, dict) and "name" in audio_prompt_file:
            audio_path = audio_prompt_file["name"]
        elif hasattr(audio_prompt_file, "name"):
            audio_path = audio_prompt_file.name  # Temp path of uploaded file
        elif isinstance(audio_prompt_file, str):
            audio_path = audio_prompt_file.strip()
        else:
            audio_path = None
    elif audio_prompt_path:  # If no upload, use input path
        audio_path = audio_prompt_path.strip()
    else:
        audio_path = None

    # Handle speech/voice prompt: file upload takes priority, then path text
    speech_prompt_value = None
    if speech_prompt_file is not None:
        if isinstance(speech_prompt_file, str) and speech_prompt_file.strip():
            speech_prompt_value = speech_prompt_file.strip()
        elif hasattr(speech_prompt_file, "name"):
            speech_prompt_value = speech_prompt_file.name
    if not speech_prompt_value and speech_prompt_path and isinstance(speech_prompt_path, str) and speech_prompt_path.strip():
        speech_prompt_value = speech_prompt_path.strip()

    if speech_prompt_value and task_type == "TTS":
        print(f"[TTS] Voice prompt: {speech_prompt_value}")

    if video_path is None and audio_path is None:
        mask_type = "mask_video_audio"
        Video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
        sync_features = torch.zeros(1, 240, 768).to(device)

    elif video_path is None:
        mask_type = "mask_video"
        Video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        sync_features = torch.zeros(1, 240, 768).to(device)
        try:
            audio_tensor = load_and_process_audio(audio_path, 16000, seconds_start, seconds_total)
        except Exception as e:
            print("Audio prompt file is empty or invalid, using zero audio tensor.")
            audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))  # Assume stereo
    elif audio_path is None:
        mask_type = "mask_audio"
        try:
            Video_tensors = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)

            sync_video_tensor = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
            sync_video=sync_transform(sync_video_tensor)
            sync_video = sync_video.unsqueeze(0).to(device)
            sync_features = _get_sync_feature_extractor().encode_video_with_sync(sync_video)
        except Exception as e:
            print("Video file is empty or invalid, using zero video tensor.")
            Video_tensors = torch.zeros((seconds_total * target_fps, 3, 224, 224))   
            sync_features = torch.zeros(1, 240, 768).to(device)         
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
    else:
        mask_type = None  # No mask needed if both provided
        try:
            Video_tensors = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)

            sync_video_tensor = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
            sync_video=sync_transform(sync_video_tensor)
            sync_video = sync_video.unsqueeze(0).to(device)
            sync_features = _get_sync_feature_extractor().encode_video_with_sync(sync_video) 

        except Exception as e:
            print("Video file is empty or invalid, using zero video tensor.")
            Video_tensors = torch.zeros((seconds_total * target_fps, 3, 224, 224))
            sync_features = torch.zeros(1, 240, 768).to(device)
        try:
            audio_tensor = load_and_process_audio(audio_path, 16000, seconds_start, seconds_total)
        except Exception as e:
            print("Audio prompt file is empty or invalid, using zero audio tensor.")
            audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))  # Assume stereo

    seconds_input=sample_size/sample_rate
    print(f'video_path: {video_path}')
    print(f'audio_path: {audio_path}')

    TASK_TTS = (task_type == "TTS")
    voice_ref_duration = 0.0

    _VOICE_PROMPT_MAX_SEC = 6.0

    if TASK_TTS and speech_prompt_value and os.path.exists(speech_prompt_value):
        voice_ref_duration = _get_audio_duration(speech_prompt_value)
        print(f"[TTS] Voice prompt audio full duration: {voice_ref_duration:.2f}s")

        if voice_ref_duration > _VOICE_PROMPT_MAX_SEC:
            print(f"[TTS] Trimming voice prompt to last {_VOICE_PROMPT_MAX_SEC}s")
            try:
                vp_wav, vp_sr = torchaudio.load(speech_prompt_value)
                keep_samples = int(_VOICE_PROMPT_MAX_SEC * vp_sr)
                vp_wav = vp_wav[..., -keep_samples:]
                import tempfile
                tmp_vp = os.path.join(tempfile.gettempdir(), f"voice_prompt_tail_{os.path.basename(speech_prompt_value)}")
                torchaudio.save(tmp_vp, vp_wav, vp_sr)
                speech_prompt_value = tmp_vp
                voice_ref_duration = _VOICE_PROMPT_MAX_SEC
                print(f"[TTS] Trimmed voice prompt saved to: {tmp_vp}")
            except Exception as e:
                print(f"[warn] voice prompt trimming failed: {e}; using full audio")

        try:
            audio_input_wav = _load_audio_input_wav(speech_prompt_value, _AUDIO_INPUT_SR, _AUDIO_INPUT_SECONDS)
            print(f"[TTS] Voice prompt audio loaded, effective duration: {voice_ref_duration:.2f}s")
        except Exception as e:
            print(f"[warn] voice prompt loading failed: {e}; use zeros")
            audio_input_wav = torch.zeros(int(_AUDIO_INPUT_SR * _AUDIO_INPUT_SECONDS), dtype=torch.float32)
            voice_ref_duration = 0.0
    else:
        try:
            audio_input_wav = _load_audio_input_wav(audio_input_path, _AUDIO_INPUT_SR, _AUDIO_INPUT_SECONDS)
        except Exception as e:
            print(f"[warn] audio_input_path processing failed: {e}; use zeros")
            audio_input_wav = torch.zeros(int(_AUDIO_INPUT_SR * _AUDIO_INPUT_SECONDS), dtype=torch.float32)

    audio_input_prompt = {
        "audio_input_wav": audio_input_wav.to(device),
        "TASK_TTS": TASK_TTS
    }
    # Use default or empty string if prompt is not provided
    if not prompt:
        prompt = ""
    # Qwen-Omni text conditioning (T2A / T2M / V2A / V2M)
    if not prompt:
        prompt = ""

    if task_type == "T2M":
        qwen_prompt = build_qwen_prompt_t2m(prompt)
    elif task_type == "V2A":
        qwen_prompt = build_qwen_prompt_v2a(prompt)
    elif task_type == "V2M":
        qwen_prompt = build_qwen_prompt_v2m(prompt)
    elif task_type == "TTS":
        qwen_prompt = build_qwen_prompt_tts(prompt)
    else:  # T2A (default)
        qwen_prompt = build_qwen_prompt_t2a(prompt)

    if audio_tensor is not None and audio_tensor.sum() > 0:
        # Insert audio marker before "<|im_end|>" in user message
        audio_tensor = np.array(audio_tensor)
        qwen_prompt = qwen_prompt.replace("<|im_end|>\n<|im_start|>assistant", f"<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n<|im_start|>assistant")

    if TASK_TTS:
        ref_text = (voice_ref_text or "").strip()
        gen_text = prompt

        # Auto ASR: if user didn't provide ref text but provided voice audio, transcribe it
        if not ref_text and speech_prompt_value and os.path.exists(speech_prompt_value):
            print(f"[TTS] No ref text provided, running ASR on voice prompt...")
            ref_text = _auto_asr_voice_prompt(speech_prompt_value, model, device)
            print(f"[TTS] ASR result: {ref_text}")

        if ref_text:
            if not ref_text.endswith(". ") and not ref_text.endswith("。"):
                if ref_text.endswith("."):
                    ref_text += " "
                else:
                    ref_text += ". "
            final_speech_prompt = ref_text + gen_text
            print(f'[TTS] ref_text: {ref_text}')
            print(f'[TTS] gen_text: {gen_text}')
            print(f'[TTS] speech_prompt (concat): {final_speech_prompt}')
        else:
            final_speech_prompt = gen_text
            print(f'[TTS] speech_prompt (no ref): {final_speech_prompt}')
    else:
        final_speech_prompt = None

    print(f'qwen_prompt: {qwen_prompt}')
    conditioning = [{
        "omni_prompt": {
            "text_prompt": qwen_prompt,
            "video_prompt": Video_tensors,
            "audio_prompt": audio_tensor,
        },
        "text_prompt": prompt,
        "audio_input_prompt": audio_input_prompt,
        "sync_feature": sync_features.to(device),
        "speech_prompt": final_speech_prompt,
    }] * batch_size

    negative_conditioning = None

    print(f"Model type: {model_type}")

    try:
        device = next(model.parameters()).device 
    except Exception as e:
        device = next(current_model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None

    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)  # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)  # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        nonlocal preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            # Use custom spectrogram generation function
            audio_spectrogram = create_spectrogram_image(denoised, sample_rate=sample_rate, figsize=(10, 3), colormap='magma')
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    # Check if long audio generation needed (T2A/T2M/V2M and duration > 10s)
    use_long_generation = (task_type in ["T2A", "T2M", "V2M"]) and (length_set > _MAX_SHORT_DURATION) and (model_type == "diffusion_cond")
    
    if use_long_generation:
        # Long audio generation: multi-segment + crossfade
        print(f"[Long Audio Generation] Task: {task_type}, Total duration: {length_set}s, using segment-based generation")
        
        segment_duration = _SEGMENT_DURATION  # 11 seconds
        overlap_duration = _OVERLAP_DURATION  # 1 second
        effective_duration = segment_duration - overlap_duration  # 10 seconds
        num_segments = int(np.ceil(length_set / effective_duration))
        fade_samples = int(overlap_duration * sample_rate)
        condition_samples = int(effective_duration * _AUDIO_INPUT_SR)  # Samples for conditioning
        
        print(f"  - Segments: {num_segments}, Segment duration: {segment_duration}s, Overlap: {overlap_duration}s")
        
        segments = []
        
        for seg_idx in range(num_segments):
            print(f"  Generating segment {seg_idx + 1}/{num_segments}...")
            
            # Compute current segment time range
            seg_start_time = seg_idx * effective_duration
            seg_end_time = min(seg_start_time + segment_duration, length_set)
            seg_actual_duration = seg_end_time - seg_start_time
            
            # Build audio condition for current segment
            if seg_idx == 0:
                # First segment: use original audio_input_wav (may be zeros)
                seg_audio_input_wav = audio_input_wav
            else:
                # Later segments: use first 10s of previous segment as condition
                prev_segment = segments[-1]
                if prev_segment.dim() == 3:
                    prev_segment = prev_segment.squeeze(0)
                if prev_segment.dim() == 2:
                    seg_audio_input_wav = prev_segment.mean(dim=0)[:condition_samples]
                else:
                    seg_audio_input_wav = prev_segment[:condition_samples]
                
                # Ensure correct length and resample to _AUDIO_INPUT_SR
                if seg_audio_input_wav.shape[0] < condition_samples:
                    pad = condition_samples - seg_audio_input_wav.shape[0]
                    seg_audio_input_wav = F.pad(seg_audio_input_wav, (0, pad), value=0.0)
                
                seg_audio_input_wav = seg_audio_input_wav.to(device)
            
            # Build conditioning for current segment
            seg_audio_input_prompt = {
                "audio_input_wav": seg_audio_input_wav,
                "TASK_TTS": TASK_TTS
            }
            
            # For V2M task, extract video frames and sync features for segment time range
            if task_type == "V2M" and video_path is not None:
                try:
                    # Extract video frames for current segment
                    seg_video_tensors = video_read_local(
                        video_path, 
                        seek_time=seconds_start + seg_start_time, 
                        duration=int(seg_actual_duration), 
                        target_fps=target_fps
                    )
                    
                    # Extract sync features for current segment
                    seg_sync_video_tensor = video_read_local(
                        video_path, 
                        seek_time=seconds_start + seg_start_time, 
                        duration=int(seg_actual_duration), 
                        target_fps=25
                    )
                    seg_sync_video = sync_transform(seg_sync_video_tensor)
                    seg_sync_video = seg_sync_video.unsqueeze(0).to(device)
                    seg_sync_features = _get_sync_feature_extractor().encode_video_with_sync(seg_sync_video)
                    
                    print(f"    Segment {seg_idx + 1}: video frames shape: {seg_video_tensors.shape}, sync features shape: {seg_sync_features.shape}")
                except Exception as e:
                    print(f"    Warning: Failed to extract video for segment {seg_idx + 1}: {e}, using zero tensors")
                    seg_video_tensors = torch.zeros((int(seg_actual_duration * target_fps), 3, 224, 224))
                    seg_sync_features = torch.zeros(1, 240, 768).to(device)
            else:
                # T2A/T2M use original video tensor (usually zeros)
                seg_video_tensors = Video_tensors
                seg_sync_features = sync_features
            
            seg_conditioning = [{
                "omni_prompt": {
                    "text_prompt": qwen_prompt,
                    "video_prompt": seg_video_tensors,
                    "audio_prompt": audio_tensor,
                },
                "text_prompt": prompt,
                "audio_input_prompt": seg_audio_input_prompt,
                "sync_feature": seg_sync_features.to(device),
                "speech_prompt": final_speech_prompt,
            }]
            
            # Generate current segment
            seg_sample_size = int(segment_duration * sample_rate)
            seg_audio = generate_diffusion_cond(
                model,
                conditioning=seg_conditioning,
                negative_conditioning=negative_conditioning,
                steps=steps,
                cfg_scale=cfg_scale,
                batch_size=1,
                sample_size=seg_sample_size,
                sample_rate=sample_rate,
                seed=seed + seg_idx if seed >= 0 else seed,  # Different seed per segment
                device=device,
                sampler_type=sampler_type,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                init_audio=None,
                init_noise_level=init_noise_level,
                mask_args=None,
                callback=None,
                scale_phi=cfg_rescale
            )
            
            segments.append(seg_audio.squeeze(0).cpu())  # [channels, samples]
            print(f"  Segment {seg_idx + 1} generated, shape: {segments[-1].shape}")
        
        # Concatenate all segments with crossfade
        print(f"  Concatenating {num_segments} segments with crossfade...")
        audio = segments[0]
        for seg_idx in range(1, num_segments):
            audio = crossfade_audio(audio, segments[seg_idx], fade_samples)
        
        # Crop to target length
        target_samples = int(length_set * sample_rate)
        if audio.shape[-1] > target_samples:
            audio = audio[..., :target_samples]
        
        print(f"  Final audio shape: {audio.shape}, duration: {audio.shape[-1] / sample_rate:.2f}s")
        
        # Adjust format for downstream processing
        audio = audio.unsqueeze(0)  # [1, channels, samples]
        
    elif model_type == "diffusion_cond":
        # Short audio generation (original logic)
        audio = generate_diffusion_cond(
            model, 
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sample_size=input_sample_size,
            sample_rate=sample_rate,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            mask_args=mask_args,
            callback=progress_callback if preview_every is not None else None,
            scale_phi=cfg_rescale
        )
    elif model_type == "diffusion_uncond":
        audio = generate_diffusion_uncond(
            model, 
            steps=steps,
            batch_size=batch_size,
            sample_size=input_sample_size,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            callback=progress_callback if preview_every is not None else None
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # For TTS with voice prompt: remove the ref-audio portion from the beginning,
    # then trim leading/trailing silence from the remaining gen-audio.
    if task_type == "TTS":
        if voice_ref_duration > 0:
            ref_samples = int(voice_ref_duration * sample_rate)
            total_samples = audio.shape[-1]
            if ref_samples < total_samples:
                print(f"[TTS] Removing voice ref portion: first {ref_samples} samples "
                      f"({voice_ref_duration:.2f}s) from {total_samples} total")
                audio = audio[..., ref_samples:]
            else:
                print(f"[TTS][warn] ref_duration ({voice_ref_duration:.2f}s) >= output length, skipping ref trim")
        audio = _trim_silence(audio, sample_rate)

    torchaudio.save("output.wav", audio, sample_rate)

    file_name = os.path.basename(video_path) if video_path else "output"

    output_dir = f"demo_result/{model_name}_prompt_{prompt[:10]}"
    output_video_path = f"{output_dir}/{file_name}"

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if video_path:
        merge_video_audio(video_path, "output.wav", output_video_path, seconds_start, seconds_total)
        
    # Generate high-quality spectrogram image
    audio_spectrogram = create_spectrogram_image(audio, sample_rate=sample_rate, figsize=(16, 4), colormap='magma')

    torch.cuda.empty_cache()
    gc.collect()

    return (output_video_path, "output.wav", audio_spectrogram)

def create_comparison_spectrogram(input_audio, output_audio, sample_rate=44100, figsize=(16, 8),
                                   colormap='magma', min_duration=10.0):
    """
    Create input/output mel spectrogram comparison (stacked vertically).
    Both panels share the same x-axis range: max(actual_duration, min_duration).
    Black background, white text — consistent with create_spectrogram_image.
    """
    try:
        import librosa
        use_librosa = True
    except ImportError:
        use_librosa = False

    hop_length = 512

    def _to_mono_np(audio):
        if isinstance(audio, torch.Tensor):
            arr = audio.detach().cpu().numpy()
        else:
            arr = np.array(audio)
        if arr.ndim > 1:
            if arr.shape[0] <= 2:
                arr = arr[0] if arr.shape[0] == 1 else np.mean(arr, axis=0)
            else:
                arr = np.mean(arr, axis=1)
        arr = arr.astype(np.float32)
        if np.max(np.abs(arr)) > 1.0:
            arr = arr / max(np.max(np.abs(arr)), 1e-10)
        return arr

    def _compute_mel_db(audio_np, sr):
        if use_librosa:
            mel = librosa.feature.melspectrogram(
                y=audio_np, sr=sr, n_fft=2048, hop_length=hop_length,
                n_mels=128, win_length=2048, center=True, power=1.0
            )
            return librosa.power_to_db(mel ** 2, ref=np.max)
        else:
            from scipy import signal
            _, _, Zxx = signal.stft(audio_np, fs=sr, nperseg=2048, noverlap=2048 - hop_length)
            mel = np.abs(Zxx)
            return 20 * np.log10(np.maximum(mel, 1e-10))

    inp_np = _to_mono_np(input_audio)
    out_np = _to_mono_np(output_audio)

    inp_mel_db = _compute_mel_db(inp_np, sample_rate)
    out_mel_db = _compute_mel_db(out_np, sample_rate)

    inp_duration = len(inp_np) / sample_rate
    out_duration = len(out_np) / sample_rate
    # Both panels use the same display range
    display_duration = max(inp_duration, out_duration, min_duration)

    # time extent of each spectrogram (columns * hop / sr)
    inp_time_extent = inp_mel_db.shape[1] * hop_length / sample_rate
    out_time_extent = out_mel_db.shape[1] * hop_length / sample_rate

    # Unified dB range (fixed 80 dB dynamic range, same as create_spectrogram_image)
    vmax = max(inp_mel_db.max(), out_mel_db.max())
    vmin = vmax - 80

    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=150)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.45)

    for ax, mel_db, time_extent, actual_dur, title in [
        (axes[0], inp_mel_db, inp_time_extent, inp_duration, f'Input  ({inp_duration:.2f}s)'),
        (axes[1], out_mel_db, out_time_extent, out_duration, f'Output  ({out_duration:.2f}s)'),
    ]:
        ax.set_facecolor('black')
        im = ax.imshow(
            mel_db,
            aspect='auto',
            origin='lower',
            cmap=colormap,
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
            extent=[0, time_extent, 0, mel_db.shape[0]],
        )
        ax.set_xlim(0, display_duration)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Mel Frequency', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.tick_params(colors='black')
        plt.colorbar(im, ax=ax, label='dB', fraction=0.02, pad=0.02)

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def generate_editing(
        editing_task_type,
        editing_sound_desc,
        editing_style_from,
        editing_style_to,
        audio_input_audio,
        audio_input_path_text,
        model_name,
        seconds_total=10,
        cfg_scale=6.0,
        steps=100,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
    ):
    """Audio Editing inference entry point"""
    global current_model_name, current_model, current_sample_rate, current_sample_size

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    _restore_model_to_cuda(current_model)

    desc = (editing_sound_desc or "").strip()
    style_from = (editing_style_from or "").strip()
    style_to = (editing_style_to or "").strip()

    if editing_task_type == "Add":
        prompt = f"Add the sound of '{desc}' to the input audio."
    elif editing_task_type == "Extract":
        prompt = f"Extract the sound of '{desc}' from the input audio."
    elif editing_task_type == "Remove":
        prompt = f"Remove the sound of '{desc}' from the input audio."
    elif editing_task_type == "Style Transfer":
        prompt = f"Change the sound of '{style_from}' to '{style_to}'."
    else:
        prompt = desc

    print(f"[Editing] Task: {editing_task_type}, Prompt: {prompt}")

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False
    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_name not in model_configurations:
        raise ValueError(f"Model {model_name} configuration not found")
    cfg = model_configurations[model_name]
    mc_path = cfg.get("model_config")
    if mc_path:
        with open(mc_path) as f:
            model_config = json.load(f)
    else:
        model_config = None
    if current_model is None or model_name != current_model_name:
        current_model, model_config, sample_rate, sample_size = load_model(
            model_name=model_name, model_config=model_config,
            model_ckpt_path=cfg.get("ckpt_path"),
            pretrained_name=cfg.get("pretrained_name"),
            pretransform_ckpt_path=cfg.get("pretransform_ckpt_path"),
            device=device)
        current_model_name = model_name
        current_sample_rate = sample_rate
        current_sample_size = sample_size
        model = current_model
    else:
        model = current_model
        sample_rate = current_sample_rate
        sample_size = current_sample_size

    # ---- Get audio input path ----
    # gr.Audio(type="filepath") returns file path string (for uploads)
    audio_input_path = None
    print(f"[Editing] audio_input_audio type={type(audio_input_audio)}, value={audio_input_audio}")
    print(f"[Editing] audio_input_path_text type={type(audio_input_path_text)}, value={audio_input_path_text}")

    if audio_input_audio is not None:
        if isinstance(audio_input_audio, str) and audio_input_audio.strip():
            audio_input_path = audio_input_audio.strip()
        elif isinstance(audio_input_audio, tuple):
            # Gradio gr.Audio(type="numpy") returns (sample_rate, numpy_array)
            # We use type="filepath", so this path should not be taken
            print(f"[Editing] audio_input_audio is tuple, unexpected for type='filepath'")
        elif isinstance(audio_input_audio, dict) and "name" in audio_input_audio:
            audio_input_path = audio_input_audio["name"]
        elif hasattr(audio_input_audio, "name"):
            audio_input_path = audio_input_audio.name

    if not audio_input_path and audio_input_path_text:
        audio_input_path = audio_input_path_text.strip()

    print(f"[Editing] Resolved audio_input_path: {audio_input_path}")

    # ---- Load audio input ----
    seconds_total = int(seconds_total)
    try:
        audio_input_wav = _load_audio_input_wav(audio_input_path, _AUDIO_INPUT_SR, _AUDIO_INPUT_SECONDS)
        print(f"[Editing] audio_input_wav loaded, shape={audio_input_wav.shape}, max={audio_input_wav.abs().max().item():.4f}")
    except Exception as e:
        print(f"[warn] audio input load failed: {e}; use zeros")
        audio_input_wav = torch.zeros(int(_AUDIO_INPUT_SR * _AUDIO_INPUT_SECONDS), dtype=torch.float32)

    audio_input_prompt = {
        "audio_input_wav": audio_input_wav.to(device),
        "TASK_TTS": False
    }

    # ---- Build conditioning ----
    qwen_prompt = build_qwen_prompt_editing(prompt)
    print(f"[Editing] qwen_prompt: {qwen_prompt}")

    try:
        device = next(model.parameters()).device
    except Exception:
        device = next(current_model.parameters()).device

    conditioning = [{
        "omni_prompt": {
            "text_prompt": qwen_prompt,
            "video_prompt": None,
            "audio_prompt": None,
        },
        "text_prompt": prompt,
        "audio_input_prompt": audio_input_prompt,
        "sync_feature": torch.zeros(1, 240, 768).to(device),
        "speech_prompt": None,
    }]

    seed = int(seed)

    # ---- Inference ----
    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=None,
        steps=int(steps),
        cfg_scale=cfg_scale,
        batch_size=1,
        sample_size=sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=None,
        init_noise_level=1.0,
        mask_args=None,
        callback=None,
        scale_phi=cfg_rescale
    )

    # ---- Post-processing ----
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # ---- Load input audio for duration reference and spectrogram ----
    inp_wav_for_spec = None
    input_duration = None
    if audio_input_path and os.path.exists(str(audio_input_path).strip()):
        try:
            inp_wav_for_spec, inp_sr = torchaudio.load(str(audio_input_path).strip())
            if inp_sr != sample_rate:
                inp_wav_for_spec = torchaudio.transforms.Resample(orig_freq=inp_sr, new_freq=sample_rate)(inp_wav_for_spec)
            input_duration = inp_wav_for_spec.shape[-1] / sample_rate
            print(f"[Editing] Input duration: {input_duration:.2f}s")
        except Exception as e:
            print(f"[warn] Failed to load input audio for duration: {e}")

    # Trim output to match input duration
    if input_duration is not None:
        input_samples = int(input_duration * sample_rate)
        if audio.shape[-1] > input_samples:
            audio = audio[..., :input_samples]
            print(f"[Editing] Output trimmed to {input_duration:.2f}s ({input_samples} samples)")

    torchaudio.save("output_editing.wav", audio, sample_rate)

    # ---- Generate comparison mel ----
    comparison_img = None
    if inp_wav_for_spec is not None:
        try:
            comparison_img = create_comparison_spectrogram(
                inp_wav_for_spec, audio, sample_rate=sample_rate, figsize=(16, 8), colormap='magma'
            )
        except Exception as e:
            print(f"[warn] Failed to generate comparison mel: {e}")

    if comparison_img is None:
        # If no input audio, show output only
        comparison_img = create_spectrogram_image(audio, sample_rate=sample_rate, figsize=(16, 4), colormap='magma')

    # ---- Prepare input audio for playback ----
    audio_input_playback = None
    if audio_input_path and os.path.exists(str(audio_input_path).strip()):
        import shutil, tempfile
        src = str(audio_input_path).strip()
        tmp_playback = os.path.join(tempfile.gettempdir(), f"editing_input_{os.path.basename(src)}")
        shutil.copy2(src, tmp_playback)
        audio_input_playback = tmp_playback
    else:
        # If no valid input audio file, save loaded wav to temp file for playback
        if audio_input_wav is not None and audio_input_wav.abs().max().item() > 0:
            try:
                tmp_input_path = "output_editing_input.wav"
                inp_save = audio_input_wav.unsqueeze(0) if audio_input_wav.dim() == 1 else audio_input_wav
                torchaudio.save(tmp_input_path, inp_save.cpu(), _AUDIO_INPUT_SR)
                audio_input_playback = tmp_input_path
            except Exception as e:
                print(f"[warn] Failed to save input audio for playback: {e}")

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    return (audio_input_playback, "output_editing.wav", comparison_img)


def create_sampling_ui(model_config_map, model_name_state):
    with gr.Blocks() as demo:
        gr.Markdown("## Generation")
        gr.Markdown(
            "Select task type to show corresponding input components.\n"
            "- **T2A / T2M**: Text prompt only\n"
            "- **V2A / V2M**: Video input (no text prompt)\n"
            "- **TTS**: Transcript + optional voice prompt for timbre cloning (auto voice-conversion when voice prompt is provided)"
        )

        with gr.Row():
            with gr.Column(scale=5):
                task_type = gr.Radio(
                    choices=["T2A", "T2M", "V2A", "V2M", "TTS"],
                    value="T2A",
                    label="Task Type",
                )

        with gr.Row() as prompt_row:
            with gr.Column():
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter your prompt (required for T2A/T2M)",
                    lines=2,
                )

        # ---- V2A / V2M: Video input ----
        with gr.Row(visible=False) as v2a_v2m_video_row:
            with gr.Column():
                gr.Markdown("**Video Input (required for V2A/V2M)**")
                video_file = gr.File(label="Upload Video File")
                video_path = gr.Textbox(label="Or Enter Video Path", placeholder="Enter video file path")

        # ---- TTS: voice prompt (optional, enables voice conversion) ----
        with gr.Row(visible=False) as tts_voice_row:
            with gr.Column():
                gr.Markdown(
                    "**Voice Prompt (optional)** — Provide a reference voice to clone its timbre.\n\n"
                    "- When provided, the output will only contain the newly generated speech (reference portion is auto-trimmed).\n"
                    "- **Voice Reference Text**: Transcript of the reference audio. "
                    "Leave empty to auto-transcribe via ASR."
                )
                speech_prompt_file = gr.Audio(
                    label="Upload or Record Voice Audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                    interactive=True,
                )
                speech_prompt_path = gr.Textbox(
                    label="Or Enter Voice Audio Path",
                    placeholder="e.g. /path/to/voice_prompt.wav",
                )
                voice_audio_preview = gr.Audio(
                    label="Voice Audio Preview",
                    interactive=False,
                    visible=False,
                )
                voice_ref_text = gr.Textbox(
                    label="Voice Reference Text (leave empty for auto ASR)",
                    placeholder="Optional: transcript of the reference audio clip.",
                )

        def _preview_voice_audio_from_path(path_str):
            import shutil, tempfile
            if path_str and isinstance(path_str, str) and path_str.strip():
                p = path_str.strip()
                if os.path.exists(p):
                    tmp_path = os.path.join(
                        tempfile.gettempdir(),
                        f"voice_preview_{os.path.basename(p)}"
                    )
                    shutil.copy2(p, tmp_path)
                    return gr.update(value=tmp_path, visible=True)
            return gr.update(value=None, visible=False)

        def _preview_voice_audio_from_upload(file_path):
            if file_path:
                return gr.update(value=None, visible=False)
            return gr.update(visible=False)

        speech_prompt_path.change(
            fn=_preview_voice_audio_from_path,
            inputs=[speech_prompt_path],
            outputs=[voice_audio_preview],
        )
        speech_prompt_file.change(
            fn=_preview_voice_audio_from_upload,
            inputs=[speech_prompt_file],
            outputs=[voice_audio_preview],
        )

        # Hidden placeholders
        sync_feature_path = gr.Textbox(visible=False, value="")
        audio_input_path = gr.Textbox(visible=False, value="")
        audio_prompt_file = gr.File(visible=False)
        audio_prompt_path = gr.Textbox(visible=False, value="")

        def _toggle_task_inputs(task):
            show_v2a_v2m = task in ["V2A", "V2M"]
            show_tts = task == "TTS"
            show_prompt = task not in ["V2A", "V2M"]
            if task == "TTS":
                prompt_update = gr.update(
                    label="Transcript (text to be spoken)",
                    placeholder="Enter the text you want the model to speak",
                )
            else:
                prompt_update = gr.update(
                    label="Text Prompt",
                    placeholder="Enter your prompt (required for T2A/T2M)",
                )
            return (
                gr.update(visible=show_v2a_v2m),
                gr.update(visible=show_tts),
                gr.update(visible=show_prompt),
                prompt_update,
            )

        task_type.change(
            fn=_toggle_task_inputs,
            inputs=[task_type],
            outputs=[v2a_v2m_video_row, tts_voice_row, prompt_row, prompt],
        )

        # Timing controls
        with gr.Row():
            with gr.Column(scale=6):
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds Start")
                seconds_total_slider = gr.Slider(minimum=1, maximum=300, step=1, value=10, label="Seconds Total")

        # Generation parameters
        with gr.Row():
            with gr.Column(scale=4):
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG Scale")

        # Sampler parameters
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Sampler Params", open=False):
                    seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")
                    sampler_type_dropdown = gr.Dropdown(
                        ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                        label="Sampler Type",
                        value="dpmpp-3m-sde"
                    )
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma Min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma Max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG Rescale Amount")

        # Init Audio parameters
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Init Audio", open=False):
                    init_audio_checkbox = gr.Checkbox(label="Use Init Audio")
                    init_audio_input = gr.Audio(label="Init Audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init Noise Level")

        # Generate button centered
        with gr.Row():
            generate_button = gr.Button("Generate", variant='primary', scale=1)

        # Output components
        with gr.Row():
            with gr.Column(scale=6):
                video_output = gr.Video(label="Output Video", interactive=False)
                audio_output = gr.Audio(label="Output Audio", interactive=False)
                audio_spectrogram_output = gr.Image(label="Output Spectrogram", type="pil", interactive=False)
                send_to_init_button = gr.Button("Send to Init Audio", scale=1)
                send_to_editing_button = gr.Button("Send to Editing", scale=1)

        # Bind send button to pass generated audio to init_audio_input
        send_to_init_button.click(
            fn=lambda audio: audio,
            inputs=[audio_output],
            outputs=[init_audio_input]
        )

        # Bind generate button
        inputs = [
            prompt,
            model_name_state,
            task_type,
            video_file,
            video_path,
            sync_feature_path,
            audio_prompt_file,
            audio_prompt_path,
            audio_input_path,
            speech_prompt_file,
            speech_prompt_path,
            voice_ref_text,
            seconds_start_slider,
            seconds_total_slider,
            cfg_scale_slider,
            steps_slider,
            preview_every_slider,
            seed_textbox,
            sampler_type_dropdown,
            sigma_min_slider,
            sigma_max_slider,
            cfg_rescale_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
        ]

        generate_button.click(
            fn=generate_cond,
            inputs=inputs,
            outputs=[
                video_output,
                audio_output,
                audio_spectrogram_output,
            ],
            api_name="generate",
        )

        return demo, audio_output, send_to_editing_button
    
def create_editing_ui(model_config_map, model_name_state):
    """Audio Editing page UI"""
    with gr.Blocks() as demo:
        gr.Markdown("## Editing")
        gr.Markdown(
            "Edit audio by selecting a task type and describing the target sound.\n"
            "- **Add**: Add a new sound to the input audio\n"
            "- **Extract**: Extract a specific sound from the input audio\n"
            "- **Remove**: Remove a specific sound from the input audio\n"
            "- **Style Transfer**: Transform one sound into another"
        )

        # Editing task selection
        with gr.Row():
            with gr.Column():
                editing_task_type = gr.Radio(
                    choices=["Add", "Extract", "Remove", "Style Transfer"],
                    value="Add",
                    label="Editing Task Type",
                )

        # Sound description input (hidden for Style Transfer)
        with gr.Row() as sound_desc_row:
            with gr.Column():
                editing_sound_desc = gr.Textbox(
                    label="Sound to Add",
                    placeholder="e.g. bird chirping, piano, wind noise ...",
                )

        # Style Transfer specific input
        with gr.Row(visible=False) as style_transfer_row:
            with gr.Column():
                editing_style_from = gr.Textbox(
                    label="Source Sound",
                    placeholder="e.g. piano"
                )
                editing_style_to = gr.Textbox(
                    label="Target Sound",
                    placeholder="e.g. guitar"
                )

        def _toggle_editing_inputs(task):
            is_style = task == "Style Transfer"
            label_map = {
                "Add": ("Sound to Add", "e.g. bird chirping, piano, wind noise ..."),
                "Extract": ("Sound to Extract", "e.g. bird chirping, speech, drums ..."),
                "Remove": ("Sound to Remove", "e.g. background noise, wind, music ..."),
            }
            label, ph = label_map.get(task, ("Sound Description", ""))
            return (
                gr.update(visible=not is_style),
                gr.update(visible=is_style),
                gr.update(label=label, placeholder=ph),
            )

        editing_task_type.change(
            fn=_toggle_editing_inputs,
            inputs=[editing_task_type],
            outputs=[sound_desc_row, style_transfer_row, editing_sound_desc],
        )

        # Audio input (supports upload, recording, path)
        with gr.Row():
            with gr.Column():
                audio_input_audio = gr.Audio(
                    label="Audio Input (upload audio file)",
                    sources=["upload"],
                    type="filepath",
                    interactive=True,
                )
                audio_input_path_text = gr.Textbox(
                    label="Or Enter Audio Input Path",
                    placeholder="Enter audio file path (lower priority than upload above)"
                )

        def _load_audio_from_path(path):
            """Copy external audio to tmp so Gradio can serve it for playback."""
            import shutil, tempfile
            if path and path.strip() and os.path.exists(path.strip()):
                src = path.strip()
                tmp_path = os.path.join(tempfile.gettempdir(), f"editing_preview_{os.path.basename(src)}")
                shutil.copy2(src, tmp_path)
                return tmp_path
            return None

        audio_input_path_text.change(fn=_load_audio_from_path, inputs=[audio_input_path_text], outputs=[audio_input_audio])
        audio_input_path_text.submit(fn=_load_audio_from_path, inputs=[audio_input_path_text], outputs=[audio_input_audio])

        # Generation parameters
        with gr.Row():
            with gr.Column(scale=6):
                seconds_total_slider = gr.Slider(minimum=1, maximum=30, step=1, value=10, label="Seconds Total")

        with gr.Row():
            with gr.Column(scale=4):
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=6.0, label="CFG Scale")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Sampler Params", open=False):
                    seed_textbox = gr.Textbox(label="Seed (-1 for random)", value="-1")
                    sampler_type_dropdown = gr.Dropdown(
                        ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                        label="Sampler Type", value="dpmpp-3m-sde"
                    )
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma Min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=1000, label="Sigma Max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG Rescale")

        # Generate button
        with gr.Row():
            generate_button = gr.Button("Generate Editing", variant='primary', scale=1)

        # Output
        with gr.Row():
            with gr.Column(scale=6):
                audio_input_preview = gr.Audio(label="Input Audio (Preview)", interactive=False)
                audio_output = gr.Audio(label="Output Audio", interactive=False)
                spectrogram_comparison = gr.Image(label="Input vs Output Spectrogram (top: Input, bottom: Output)", type="pil")

        # Bind generate
        generate_button.click(
            fn=generate_editing,
            inputs=[
                editing_task_type,
                editing_sound_desc,
                editing_style_from,
                editing_style_to,
                audio_input_audio,
                audio_input_path_text,
                model_name_state,
                seconds_total_slider,
                cfg_scale_slider,
                steps_slider,
                seed_textbox,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                cfg_rescale_slider,
            ],
            outputs=[audio_input_preview, audio_output, spectrogram_comparison],
            api_name="generate_editing"
        )

    return demo, audio_input_audio

    
def create_understanding_ui(model_name_state):
    """Understanding UI: multi-turn chat with optional audio / video per turn."""
    with gr.Blocks() as demo:
        gr.Markdown("## Understanding")
        gr.Markdown(
            "Multi-turn conversation with the model. "
            "Each turn can freely combine **Text / Audio / Video** inputs. "
            "Text history is preserved across turns for context."
        )

        chat_history_state = gr.State([])
        chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter your question or description (can be used alone or combined with audio/video)",
                    lines=3,
                )

        with gr.Row():
            with gr.Column():
                audio_input_file = gr.Audio(
                    label="Audio Input (optional, upload takes priority)",
                    sources=["upload"],
                    type="filepath",
                    interactive=True,
                )
                audio_input_path_text = gr.Textbox(
                    label="Or Enter Audio Path",
                    placeholder="e.g. /path/to/audio.wav",
                )
            with gr.Column():
                video_input_file = gr.Video(
                    label="Video Input (optional, upload takes priority)",
                    interactive=True,
                )
                video_input_path_text = gr.Textbox(
                    label="Or Enter Video Path",
                    placeholder="e.g. /path/to/video.mp4",
                )
                use_audio_in_video = gr.Checkbox(
                    label="Use Audio Track in Video",
                    value=False,
                )

        with gr.Row():
            with gr.Column():
                av_start_time = gr.Number(
                    label="Start Time (s) — for audio / video",
                    value=0.0,
                    minimum=0.0,
                    step=0.5,
                )
            with gr.Column():
                av_duration = gr.Number(
                    label="Duration (s) — 0 = use full clip",
                    value=0.0,
                    minimum=0.0,
                    step=0.5,
                )

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    system_prompt = gr.Textbox(
                        label="System Prompt (optional)",
                        placeholder="Leave empty for default system prompt",
                        lines=2,
                    )

        with gr.Row():
            understand_button = gr.Button("Send", variant="primary", scale=1)
            clear_button = gr.Button("Clear History", scale=1)

        understand_button.click(
            fn=generate_understanding,
            inputs=[
                model_name_state,
                text_input,
                audio_input_file,
                audio_input_path_text,
                video_input_file,
                video_input_path_text,
                av_start_time,
                av_duration,
                system_prompt,
                use_audio_in_video,
                chat_history_state,
            ],
            outputs=[chatbot, chat_history_state, text_input],
            api_name="understand",
        )

        def _clear_chat():
            return [], [], ""

        clear_button.click(
            fn=_clear_chat,
            inputs=[],
            outputs=[chatbot, chat_history_state, text_input],
        )

    return demo


def create_txt2audio_ui(model_config_map, default_model_name):
    with gr.Blocks() as ui:
        model_name_state = gr.State(default_model_name)

        with gr.Tab("Understanding"):
            create_understanding_ui(model_name_state)
        with gr.Tab("Generation"):
            _, gen_audio_output, send_to_editing_btn = create_sampling_ui(model_config_map, model_name_state)
        with gr.Tab("Editing"):
            _, editing_audio_input = create_editing_ui(model_config_map, model_name_state)

        send_to_editing_btn.click(
            fn=lambda audio: audio,
            inputs=[gen_audio_output],
            outputs=[editing_audio_input],
        )
    return ui

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global model_configurations
    global device

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_configs = json.load(f)
    else:
        model_configs = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Assume model_configs is a dict containing multiple model configs
    if isinstance(model_configs, dict):
        model_options = model_configs
    elif isinstance(model_configs, list):
        # If list, each element is a model config
        model_options = {f"model_{i}": cfg for i, cfg in enumerate(model_configs)}
    else:
        model_options = {"default": {"model_config": model_config_path, "ckpt_path": ckpt_path, "pretrained_name": pretrained_name}}

    # Load all model configs but do not load models yet
    for model_name, cfg in model_options.items():
        model_config = cfg.get("model_config")
        ckpt_path = cfg.get("ckpt_path")
        pretrained_name = cfg.get("pretrained_name")
        pretransform_ckpt_path = cfg.get("pretransform_ckpt_path")
        model_type = cfg.get("model_type", "diffusion_cond")  # Default model type

        model_configurations[model_name] = {
            "model_config": model_config,
            "ckpt_path": ckpt_path,
            "pretrained_name": pretrained_name,
            "pretransform_ckpt_path": pretransform_ckpt_path,
            "model_type": model_type
        }

    default_model_name = list(model_configurations.keys())[0]
    ui = create_txt2audio_ui(model_configurations, default_model_name)
    return ui

