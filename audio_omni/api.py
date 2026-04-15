"""
High-level API for Audio-Omni inference.

Usage:
    from audio_omni.api import AudioOmni

    model = AudioOmni("model/Audio-Omni.json", "model/model.ckpt")
    model.generate("T2A", prompt="A dog barking in a park")
    model.understand(audio="example.wav", question="Describe the sounds.")
"""

import json
import math
import os
import typing as tp

import torch
import torchaudio
from einops import rearrange
from torchvision import transforms
from torchvision.transforms import v2

from .models.factory import create_model_from_config
from .models.utils import load_ckpt_state_dict
from .inference.generation import generate_diffusion_cond
from .prompts import (
    build_prompt,
    EDITING_ADD, EDITING_REMOVE, EDITING_EXTRACT, EDITING_STYLE,
)

_SYNC_SIZE = 224
_sync_transform = v2.Compose([
    v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(_SYNC_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class AudioOmni:
    """Unified interface for Audio-Omni generation, editing, and understanding."""

    def __init__(
        self,
        config_path: str = "model/Audio-Omni.json",
        ckpt_path: str = "model/model.ckpt",
        device: str = "cuda",
    ):
        self._config_path = config_path
        with open(config_path) as f:
            self.model_config = json.load(f)

        self.model = create_model_from_config(self.model_config)
        self.model.load_state_dict(load_ckpt_state_dict(ckpt_path))
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()

        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]

    # ------------------------------------------------------------------
    # Generation (T2A, T2M, TTS, V2A, V2M)
    # ------------------------------------------------------------------
    def generate(
        self,
        task: str,
        prompt: str = "",
        *,
        video_path: tp.Optional[str] = None,
        voice_prompt_path: tp.Optional[str] = None,
        voice_ref_text: tp.Optional[str] = None,
        steps: int = 100,
        cfg_scale: float = 7.0,
        seconds_total: int = 10,
        seed: int = -1,
    ) -> torch.Tensor:
        """Generate audio.

        Args:
            task: "T2A", "T2M", "V2A", "V2M", or "TTS".
            prompt: Text description (T2A/T2M) or transcript (TTS).
            video_path: Path to video file (V2A/V2M).
            voice_prompt_path: Reference voice audio for TTS voice cloning.
            voice_ref_text: Transcript of the reference voice audio.
            steps: Number of diffusion steps.
            cfg_scale: Classifier-free guidance scale.
            seconds_total: Duration of generated audio in seconds.
            seed: Random seed (-1 for random).

        Returns:
            Audio tensor of shape (channels, samples), int16, at model sample_rate.
        """
        qwen_prompt = build_prompt(task, prompt)
        conditioning = self._build_conditioning(
            qwen_prompt=qwen_prompt,
            prompt=prompt,
            task=task,
            video_path=video_path,
            voice_prompt_path=voice_prompt_path,
            voice_ref_text=voice_ref_text,
            seconds_total=seconds_total,
        )

        output = generate_diffusion_cond(
            self.model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sample_rate=self.sample_rate,
            seed=seed,
            device=self.device,
        )
        return self._postprocess(output)

    # ------------------------------------------------------------------
    # Editing (Add, Remove, Extract, Style Transfer)
    # ------------------------------------------------------------------
    def edit(
        self,
        task: str,
        source_audio: str,
        *,
        desc: tp.Optional[str] = None,
        source_category: tp.Optional[str] = None,
        target_category: tp.Optional[str] = None,
        steps: int = 100,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> torch.Tensor:
        """Edit audio.

        Args:
            task: "Add", "Remove", "Extract", or "Style Transfer".
            source_audio: Path to the source audio file.
            desc: Sound object description (for Add/Remove/Extract).
            source_category: Source sound category (for Style Transfer).
            target_category: Target sound category (for Style Transfer).
            steps: Number of diffusion steps.
            cfg_scale: Classifier-free guidance scale.
            seed: Random seed (-1 for random).

        Returns:
            Audio tensor of shape (channels, samples), int16, at model sample_rate.
        """
        if task == "Style Transfer":
            prompt = EDITING_STYLE.format(source=source_category, target=target_category)
            qwen_prompt = build_prompt("Style Transfer", source=source_category, target=target_category)
        else:
            prompt = {"Add": EDITING_ADD, "Remove": EDITING_REMOVE, "Extract": EDITING_EXTRACT}[task].format(desc=desc)
            qwen_prompt = build_prompt(task, desc=desc)

        source_wav = self._load_audio_mono(source_audio)
        audio_input_prompt = {"audio_input_wav": source_wav.to(self.device), "TASK_TTS": False}

        conditioning = [{
            "omni_prompt": {"text_prompt": qwen_prompt, "video_prompt": None, "audio_prompt": None},
            "text_prompt": prompt,
            "audio_input_prompt": audio_input_prompt,
            "sync_feature": torch.zeros(1, 240, 768, device=self.device),
            "speech_prompt": None,
        }]

        output = generate_diffusion_cond(
            self.model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sample_rate=self.sample_rate,
            seed=seed,
            device=self.device,
        )
        return self._postprocess(output)

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------
    def understand(
        self,
        question: str,
        *,
        audio: tp.Optional[str] = None,
        video: tp.Optional[str] = None,
    ) -> str:
        """Understand audio/video content.

        Args:
            question: Question about the audio/video.
            audio: Path to audio file (optional).
            video: Path to video file (optional).

        Returns:
            Text response from the model.
        """
        from .prompts import SYSTEM_UNDERSTANDING

        omni_cond = self._get_omni_conditioner()
        processor = omni_cond.processor
        omni_model = omni_cond.model

        user_content = []
        if video:
            user_content.append({"type": "video", "video": video})
        if audio:
            user_content.append({"type": "audio", "audio": audio})
        user_content.append({"type": "text", "text": question})

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_UNDERSTANDING}]},
            {"role": "user", "content": user_content},
        ]

        from qwen_omni_utils import process_mm_info
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(omni_model.device).to(omni_model.dtype)

        text_ids = omni_model.generate(**inputs, return_audio=False, use_audio_in_video=True)
        return processor.batch_decode(text_ids, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_omni_conditioner(self):
        conditioner_wrapper = getattr(self.model, "conditioner", None)
        if conditioner_wrapper is None:
            raise RuntimeError("Model does not contain a conditioner.")
        for _key, cond in conditioner_wrapper.conditioners.items():
            cls_name = type(cond).__name__
            if "OmniConditioner" in cls_name or "QwenOmni" in cls_name:
                return cond
        raise RuntimeError("Model does not contain an OmniConditioner.")

    def _build_conditioning(self, qwen_prompt, prompt, task, video_path, voice_prompt_path, voice_ref_text, seconds_total):
        target_fps = self.model_config.get("video_fps", 5)
        sample_rate = self.sample_rate

        speech_prompt = None
        if task == "TTS":
            ref_text = (voice_ref_text or "").strip()
            if ref_text:
                if not ref_text.endswith(". ") and not ref_text.endswith("。"):
                    ref_text = ref_text.rstrip(".") + ". "
                speech_prompt = ref_text + prompt
            else:
                speech_prompt = prompt

        if voice_prompt_path and os.path.exists(voice_prompt_path):
            voice_wav = self._load_audio_mono(voice_prompt_path, max_seconds=6)
        else:
            voice_wav = torch.zeros(44100 * seconds_total, dtype=torch.float32)
        audio_input_prompt = {"audio_input_wav": voice_wav.to(self.device), "TASK_TTS": task == "TTS"}

        video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
        sync_features = torch.zeros(1, 240, 768, device=self.device)

        if video_path and os.path.exists(video_path):
            try:
                video_tensors = self._read_video(video_path, seek_time=0., duration=seconds_total, target_fps=target_fps)
            except Exception as e:
                print(f"[warn] Video frame extraction failed: {e}; using zero video tensors")
                video_tensors = torch.zeros(int(seconds_total * target_fps), 3, 224, 224)

            try:
                sync_features = self._extract_sync_features(video_path, seconds_total)
            except Exception as e:
                print(f"[warn] Sync feature extraction failed: {e}; using zero sync features")
                sync_features = torch.zeros(1, 240, 768, device=self.device)

        return [{
            "omni_prompt": {"text_prompt": qwen_prompt, "video_prompt": video_tensors, "audio_prompt": audio_tensor},
            "text_prompt": prompt,
            "audio_input_prompt": audio_input_prompt,
            "sync_feature": sync_features.to(self.device),
            "speech_prompt": speech_prompt,
        }]

    def _load_audio_mono(self, path: str, max_seconds: int = 10) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != 44100:
            wav = torchaudio.transforms.Resample(sr, 44100)(wav)
        wav = wav.mean(0)[:44100 * max_seconds]
        return wav

    def _read_video(self, filepath: str, seek_time: float = 0., duration: int = 10, target_fps: int = 5) -> torch.Tensor:
        from decord import VideoReader, cpu as decord_cpu
        vr = VideoReader(filepath, ctx=decord_cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        seek_frame = int(seek_time * fps)
        total_frames_to_read = int(target_fps * duration)
        frame_interval = int(math.ceil(fps / target_fps))
        end_frame = min(seek_frame + total_frames_to_read * frame_interval, total_frames)
        frame_ids = list(range(seek_frame, end_frame, frame_interval))
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        if frames.shape[2] != 224 or frames.shape[3] != 224:
            frames = transforms.Resize((224, 224))(frames)
        target_n = duration * target_fps
        if frames.shape[0] > target_n:
            frames = frames[:target_n]
        elif frames.shape[0] < target_n:
            last = frames[-1:]
            frames = torch.cat([frames, last.repeat(target_n - frames.shape[0], 1, 1, 1)], dim=0)
        return frames

    def _extract_sync_features(self, video_path: str, duration: int = 10) -> torch.Tensor:
        from mmaudio.model.utils.features_utils import FeaturesUtils
        from mmaudio.utils.download_utils import download_model_if_needed
        from pathlib import Path

        if not hasattr(self, '_sync_extractor') or self._sync_extractor is None:
            synchformer_ckpt = os.environ.get("SYNCHFORMER_CKPT", "")
            if not synchformer_ckpt or not os.path.exists(synchformer_ckpt):
                synchformer_ckpt = os.path.join(
                    os.path.dirname(self._config_path), "synchformer_state_dict.pth"
                )
            if not os.path.exists(synchformer_ckpt):
                download_model_if_needed(Path(synchformer_ckpt))
            self._sync_extractor = FeaturesUtils(
                tod_vae_ckpt=None,
                enable_conditions=True,
                synchformer_ckpt=synchformer_ckpt,
                mode='16k',
            ).eval().to(self.device)
        sync_video = self._read_video(video_path, seek_time=0., duration=duration, target_fps=25)
        sync_video = _sync_transform(sync_video)
        sync_video = sync_video.unsqueeze(0).to(self.device)
        return self._sync_extractor.encode_video_with_sync(sync_video)

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        return output
