import os
import torch
import typing as tp

from torch import nn
import einops
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.scale = torch.ones(hidden_size)
        self.eps = eps
        
    def forward(self, x):
        mean_sq = torch.mean(x**2, dim=-1, keepdim=True)
        scale = torch.rsqrt(mean_sq + self.eps)
        return x * scale

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class OmniConditioner(Conditioner):
    QWEN_VL_MODELS = ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-3B"]
    def __init__(self,
                 output_dim: int,
                 qwen_omni_model_name: str = "Qwen/Qwen2.5-Omni-3B",
                 add_sync_feature: bool = True,
                 audio_encoder_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
                 video_fps: int = 5,
                 project_out: bool = False,
                 in_features: int = 1619,
                 out_features: int = 128,
                 pad_length: int = 4096,
                 zip_features: bool = False,
                 zip_length: int = 512,
                 layer_idx: int = -1,
                 connector: bool = False,
                 connector_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
                 ):
        assert qwen_omni_model_name in self.QWEN_VL_MODELS, f"Unknown QwenVL model name: {qwen_omni_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)

        self.qwen_omni_model_name = qwen_omni_model_name
        self.video_fps = video_fps
        self.layer_idx = layer_idx

        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.pad_length = pad_length


        if zip_features:
            self.zip_features = True
            self.proj_omni_seq_len = nn.Linear(in_features=pad_length, out_features=zip_length)

        _qwen_model_path = os.environ.get(
            "QWEN_OMNI_MODEL_PATH",
            qwen_omni_model_name,
        )
        _skip_qwen_weights = os.environ.get("SKIP_QWEN_WEIGHTS", "0").strip().lower() in ("1", "true", "yes")

        if _skip_qwen_weights:
            from transformers import AutoConfig
            _cfg_src = _qwen_model_path if os.path.isdir(_qwen_model_path) else qwen_omni_model_name
            qwen_config = AutoConfig.from_pretrained(_cfg_src, trust_remote_code=True)
            self.model = Qwen2_5OmniForConditionalGeneration._from_config(qwen_config)
            self.model.to(dtype=torch.bfloat16)
            self.processor = Qwen2_5OmniProcessor.from_pretrained(_cfg_src)
            print("qwen-omni: skeleton created (SKIP_QWEN_WEIGHTS=1), weights will come from ckpt.")
        else:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(_qwen_model_path)
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                _qwen_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print("qwen-omni loaded from pretrained weights.")

        self.videos_kwargs = {
            "do_resize": True,
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
        }

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if os.environ.get("QWEN_KEEP_AUX_MODULES", "0").strip().lower() in ("1", "true", "yes", "y", "on"):
            for name, module in self.model.named_children():
                if name != "thinker":
                    module.cpu()
            print("qwen-omni: only thinker kept on GPU, other components moved to CPU.")
        else:
            for name in list(self.model._modules.keys()):
                if name != "thinker":
                    self.model._modules.pop(name, None)
            print("qwen-omni: only thinker kept; other components removed to reduce ckpt size.")

        self.in_features = in_features
        self.out_features = out_features
        if qwen_omni_model_name == "Qwen/Qwen2.5-Omni-7B":
            qwen_feature_dim = 2048
        elif qwen_omni_model_name == "Qwen/Qwen2.5-Omni-3B":
            qwen_feature_dim = 2048
        
        self.proj_features = nn.Linear(in_features=qwen_feature_dim, out_features=768)

        # Initialize weights to 0
        nn.init.constant_(self.proj_features.weight, 0)
        nn.init.constant_(self.proj_features.bias, 0)


        self.norm = RMSNorm(qwen_feature_dim)

    def forward(self, prompt_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        pad_length = self.pad_length
        batch_size = len(prompt_list)
        if isinstance(prompt_list[0], dict):

            text_prompts =  [item.get("text_prompt") for item in prompt_list]
            video_prompts = [item.get("video_prompt") for item in prompt_list]
            audio_prompts = [item.get("audio_prompt") for item in prompt_list]
            
            # Remove None from the list; if the list is all None or empty, pass None to the processor
            text_prompts = [p for p in text_prompts if p is not None] or None
            # For video_prompts, additional checks needed: filter out None and invalid values like empty lists/strings
            video_prompts_filtered = []
            for p in video_prompts:
                if p is not None and p.sum() != 0:
                    p = p.to('cpu')
                    # Check if it is valid video data (not empty list, empty string, etc.)
                    if isinstance(p, (list, tuple)) and len(p) > 0:
                        video_prompts_filtered.append(p)
                    elif isinstance(p, torch.Tensor) and p.numel() > 0:
                        video_prompts_filtered.append(p)
                    elif isinstance(p, str) and p.strip():
                        video_prompts_filtered.append(p)
                    elif not isinstance(p, (list, tuple, str, torch.Tensor)):
                        # Keep other types (e.g., numpy array) as well
                        video_prompts_filtered.append(p)
            video_prompts = video_prompts_filtered if video_prompts_filtered else None
            audio_prompts = [p for p in audio_prompts if p is not None and p.sum() > 0] or None
            
            inputs = self.processor(text=text_prompts, audio=audio_prompts, images=None, videos=video_prompts,  return_tensors="pt", padding=True, use_audio_in_video=False)
            qwen_mask = inputs.get("attention_mask", None)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision, use float32
                    qwen_omni_output = self.model.thinker(**inputs,output_hidden_states=True,return_dict=True)
                    omni_features = qwen_omni_output.hidden_states[self.layer_idx].bfloat16() # omni_features: [B, seq_len, hidden_dim]

            if qwen_mask is not None:
                qwen_mask = qwen_mask.to(device)
                # Ensure sequence lengths match
                seq_len = omni_features.shape[1]
                mask_seq_len = qwen_mask.shape[1]
                if mask_seq_len != seq_len:
                    if mask_seq_len > seq_len:
                        qwen_mask = qwen_mask[:, :seq_len]
                    else:
                        pad_length_mask = seq_len - mask_seq_len
                        qwen_mask = F.pad(qwen_mask, (0, pad_length_mask), value=0)
                
                # Extract real features: keep only positions where mask=1
                batch_size = omni_features.shape[0]
                real_features_list = []
                real_lengths = []
                
                for b in range(batch_size):
                    batch_mask = qwen_mask[b].bool()  # [seq_len]
                    batch_features = omni_features[b]  # [seq_len, hidden_dim]
                    # Keep only features at real positions (remove padding)
                    real_features = batch_features[batch_mask]  # [real_len, hidden_dim]
                    real_features_list.append(real_features)
                    real_lengths.append(real_features.shape[0])
                # Directly truncate/pad each real_features to pad_length
                padded_features_list = []
                for real_features in real_features_list:
                    real_len = real_features.shape[0]

                    # If real length exceeds pad_length, use interpolation to compress (instead of direct truncation)
                    if real_len > pad_length:
                        print(f"[OmniConditioner] Interpolating real_len {real_len} to pad_length {pad_length}")
                        real_features = real_features.transpose(0, 1)  # [hidden_dim, real_len]
                        real_features = F.interpolate(
                            real_features.unsqueeze(0),  # [1, hidden_dim, real_len]
                            size=pad_length,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0)  # [hidden_dim, pad_length]
                        real_features = real_features.transpose(0, 1)  # [pad_length, hidden_dim]
                        real_len = pad_length

                    # Right-pad to pad_length (if insufficient)
                    if real_len < pad_length:
                        padded_features = F.pad(real_features, (0, 0, 0, pad_length - real_len), value=0)
                    else:
                        padded_features = real_features

                    padded_features_list.append(padded_features)
                # Re-stack into a tensor
                omni_features = torch.stack(padded_features_list, dim=0)  # [B, pad_length, hidden_dim]
            else:
                # If no qwen_mask, use all features
                orig_len = omni_features.shape[1]
                real_lengths = [orig_len] * omni_features.shape[0]
                try:
                    assert pad_length >= orig_len, f"pad_length ({pad_length}) must be >= omni_features length ({orig_len})"
                except:
                    print(f"pad_length ({pad_length}) must be >= omni_features length ({orig_len})")
                # Pad omni_features along the time dimension to pad_length (right padding)
                omni_features = F.pad(omni_features, (0, 0, 0, pad_length - orig_len), value=0)

            
            if qwen_mask is not None:
                # Construct mask based on real lengths
                attention_mask = torch.zeros(omni_features.shape[0], pad_length, dtype=torch.bool, device=omni_features.device)
                for b, real_len in enumerate(real_lengths):
                    attention_mask[b, :real_len] = True
            else:
                # If no qwen_mask, create an all-True mask
                attention_mask = torch.ones(omni_features.shape[0], orig_len, dtype=torch.bool, device=omni_features.device)
                attention_mask = F.pad(attention_mask, (0, pad_length - orig_len), value=False)


        elif isinstance(prompt_list[0], torch.Tensor):

            padded_features_list = []
            real_lengths = []

            for feature_tensor in prompt_list:
                feature_tensor = feature_tensor.to(device)
                # Unify to [seq_len, hidden_dim] format (reduce dimension conversions)
                if feature_tensor.dim() == 3:
                    feature_tensor = feature_tensor[0]  # Direct indexing, faster than squeeze

                orig_len = feature_tensor.shape[0]
                real_len = min(orig_len, pad_length)
                real_lengths.append(real_len)
                
                # Directly pad/truncate to pad_length (in one step)
                if orig_len < pad_length:
                    padded_feature = F.pad(feature_tensor, (0, 0, 0, pad_length - orig_len), value=0)
                elif orig_len > pad_length:
                    padded_feature = feature_tensor[:pad_length]
                else:
                    padded_feature = feature_tensor
                
                padded_features_list.append(padded_feature)

            # Batch stack: [B, pad_length, hidden_dim]
            omni_features = torch.stack(padded_features_list, dim=0)

            # Vectorized attention_mask creation (avoid Python loops, use GPU acceleration)
            seq_indices = torch.arange(pad_length, device=omni_features.device, dtype=torch.long).unsqueeze(0)  # [1, pad_length]
            real_lengths_tensor = torch.tensor(real_lengths, device=omni_features.device, dtype=torch.long).unsqueeze(1)  # [B, 1]
            attention_mask = seq_indices < real_lengths_tensor  # [B, pad_length] - vectorized operation
        else:
            raise ValueError(f"Unknown prompt type: {type(prompt_list[0])}")


        omni_features = self.norm(omni_features)
        omni_features = self.proj_features(omni_features.to(self.proj_features.weight.dtype))

        if hasattr(self, "zip_features") and self.zip_features:
            omni_features = self.proj_omni_seq_len(omni_features.transpose(1, 2)).transpose(1, 2)

        return omni_features, attention_mask


class SynchformerConditioner(Conditioner):
    def __init__(self, sync_fps: int = 32, sync_seq_dim: int = 240, sync_output_dim: int = 128, input_dim: int=768, output_dim: int = 1536, project_out: bool = True):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        self.dim_project = nn.Linear(sync_seq_dim, sync_output_dim)
        self.empty_sync_feature = torch.zeros(1, 240, 768)

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)
        torch.nn.init.constant_(self.proj.weight, 0.)
        torch.nn.init.constant_(self.proj.bias, 0.)

    def forward(self, Video_sync_features: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.empty_sync_feature = self.empty_sync_feature.to(device=device, dtype=Video_sync_features[0].dtype)

        Video_sync_features = torch.cat(Video_sync_features, dim=0).to(device)
        # Detect which batches have all-zero values
        is_zero_batch = (Video_sync_features.abs().sum(dim=(1, 2)) == 0)  # shape: (b,)
        
        # Replace all-zero batches with empty_sync_feature
        Video_sync_features[is_zero_batch] = self.empty_sync_feature

        Video_sync_features = self.norm(Video_sync_features)
        Video_sync_features = self.proj(Video_sync_features)


        Video_sync_features = einops.rearrange(Video_sync_features, 'b t c -> b c t')
        Video_sync_features = self.dim_project(Video_sync_features)
        Video_sync_features = einops.rearrange(Video_sync_features, 'b c t -> b t c')
        return Video_sync_features, torch.ones(Video_sync_features.shape[0], Video_sync_features.shape[1]).to(device)

                   

# Text embedding
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)

def exists(v):
    return v is not None

def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):  # noqa: F722
        text = text.to(torch.long)
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text

import jieba
from pypinyin import Style, lazy_pinyin
import torchaudio

class TTSConditioner(Conditioner):

    TTS_MODELS = ["tts-f5", "tts-f5-token-level", "tts-f5-qwen-tokenizer", "tts-qwen-emb_qwen-tokenizer"]
    
    TTS_MODEL_DIMS = {
        "tts-f5": 768,
        "tts-f5-token-level": 768,
        "tts-f5-qwen-tokenizer": 768,
        "tts-qwen-emb_qwen-tokenizer": 768
    }

    def __init__(
            self,
            output_dim: int,
            tts_model_name: str = "tts-f5",
            vocab_file: str = './vocab.txt',
            seq_len: int=2584,
            project_out: bool = False,
            proj_seq_len: int = 2584,
    ):
        assert tts_model_name in self.TTS_MODELS, f"Unknown TTS model name: {tts_model_name}"
        super().__init__(self.TTS_MODEL_DIMS[tts_model_name], output_dim, project_out=project_out)
        self.tts_model_name = tts_model_name

        if self.tts_model_name == "tts-f5":
            tokenizer = "custom"
            print("\nvocab : ", vocab_file)
            print("token : ", tokenizer)
            self.vocab_char_map, vocab_size = self.get_tokenizer(vocab_file, tokenizer)
            self.seq_len = seq_len

            print(f'vocab_size: {vocab_size}')
            text_num_embeds=vocab_size
            text_dim = 512
            text_mask_padding = True
            conv_layers = 4

            self.text_embed = TextEmbedding(
                text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
            )
            self.proj_features = nn.Linear(in_features=512, out_features=768)
            if proj_seq_len != seq_len:
                self.proj_seq_len = nn.Linear(in_features=seq_len, out_features=proj_seq_len)
            else:
                self.proj_seq_len = nn.Identity()
            self.empty_speech_feat = nn.Parameter(torch.zeros(1, proj_seq_len, 768), requires_grad=True)
            nn.init.constant_(self.empty_speech_feat, 0)  

            self.norm_features = RMSNorm(768)
            self.norm = RMSNorm(768)


    def get_tokenizer(self, dataset_name, tokenizer: str = "pinyin"):
        """
        tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                    - "char" for char-wise tokenizer, need .txt vocab_file
                    - "byte" for utf-8 tokenizer
                    - "custom" if you're directly passing in a path to the vocab.txt you want to use
        vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                    - if use "char", derived from unfiltered character & symbol counts of custom dataset
                    - if use "byte", set to 256 (unicode byte range)
        """
        if tokenizer in ["pinyin", "char"]:
            tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                vocab_char_map = {}
                for i, char in enumerate(f):
                    vocab_char_map[char[:-1]] = i
            vocab_size = len(vocab_char_map)
            assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

        elif tokenizer == "byte":
            vocab_char_map = None
            vocab_size = 256

        elif tokenizer == "custom":
            with open(dataset_name, "r", encoding="utf-8") as f:
                vocab_char_map = {}
                for i, char in enumerate(f):
                    vocab_char_map[char[:-1]] = i
            vocab_size = len(vocab_char_map)

        return vocab_char_map, vocab_size

    def convert_char_to_pinyin(self, text_list, polyphone=True):
        
        if jieba.dt.initialized is False:
            jieba.default_logger.setLevel(50)  # CRITICAL
            jieba.initialize()

        final_text_list = []
        custom_trans = str.maketrans(
            {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        )  # add custom trans here, to address oov

        def is_chinese(c):
            return (
                "\u3100" <= c <= "\u9fff"  # common chinese characters
            )

        for text in text_list:
            char_list = []
            text = text.translate(custom_trans)
            for seg in jieba.cut(text):
                seg_byte_len = len(bytes(seg, "UTF-8"))
                if seg_byte_len == len(seg):  # if pure alphabets and symbols
                    if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                        char_list.append(" ")
                    char_list.extend(seg)
                elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                    seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                    for i, c in enumerate(seg):
                        if is_chinese(c):
                            char_list.append(" ")
                        char_list.append(seg_[i])
                else:  # if mixed characters, alphabets and symbols
                    for c in seg:
                        if ord(c) < 256:
                            char_list.extend(c)
                        elif is_chinese(c):
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
            final_text_list.append(char_list)

        return final_text_list

    def convert_char_to_text(self, text_list, polyphone=True):
        
        if jieba.dt.initialized is False:
            jieba.default_logger.setLevel(50)  # CRITICAL
            jieba.initialize()

        final_text_list = []
        custom_trans = str.maketrans(
            {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        )  # add custom trans here, to address oov


        for text in text_list:
            char_list = []
            text = text.translate(custom_trans)
            final_text_list.append(text)

        return final_text_list

    # char tokenizer, based on custom dataset's extracted .txt file
    def list_str_to_idx(
        self,
        text,
        vocab_char_map: dict[str, int],  # {char: idx}
        padding_value=-1,
    ) -> int:  # noqa: F722
        list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
        text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
        return text

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)


        # Record which texts are None
        is_none = [text is None for text in texts]
        # If all are None, directly return empty_speech_feat
        if all(is_none):
            empty_speech_feat = self.empty_speech_feat.expand(batch_size, -1, -1).to(device)
            return empty_speech_feat, torch.ones(empty_speech_feat.shape[0], empty_speech_feat.shape[1]).to(device)
        
        # Replace None with empty string for processing
        texts_processed = [text if text is not None else "" for text in texts]
        
        final_text_list = self.convert_char_to_pinyin(texts_processed)
        
        if isinstance(final_text_list, list):
            if self.tts_model_name == "tts-f5":
                text = self.list_str_to_idx(final_text_list, self.vocab_char_map).to(device)

        if text.dtype != torch.long:
            text = text.long()
        
        text_embed = self.text_embed(text, self.seq_len, drop_text=False)
        text_embed = self.norm_features(text_embed)
        text_embed = self.proj_features(text_embed)
        
        text_embed = self.proj_seq_len(text_embed.transpose(1, 2)).transpose(1, 2)
        # For None texts, replace with empty_speech_feat
        empty_speech_feat = self.empty_speech_feat.expand(batch_size, -1, -1).to(device)
        is_none_tensor = torch.tensor(is_none, device=device).view(batch_size, 1, 1)
        text_embed = torch.where(is_none_tensor, empty_speech_feat, text_embed)
        
        text_embed = self.norm(text_embed)
        return text_embed, torch.ones(text_embed.shape[0], text_embed.shape[1]).to(device)


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    org_device = waveform.device
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(org_device)
    
    if len(waveform.shape) == 3:
        waveform = waveform.mean(dim=1, keepdim=True)  # 'b c nw -> b 1 nw'
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'
    
    assert len(waveform.shape) == 2
    waveform = waveform.to(torch.float32)
    try:
        mel = mel_stft(waveform)
    except:
        mel_stft = mel_stft.to('cpu')
        mel = mel_stft(waveform.to('cpu'))
        mel = mel.to(waveform.device)
    mel = mel.to(org_device)

    mel = mel.clamp(min=1e-5).log()
    return mel


class AudioMelConditioner(Conditioner):

    Mel_MODELS = ["mel_features", "vocos"]
    
    MEL_MODEL_DIMS = {
        "vocos": 768,
        "mel_features": 768
    }
    def __init__(
            self,
            output_dim: int,
            mel_spec_type: str = "mel_features",
            n_fft: int = 1024,
            hop_length: int = 256,
            win_length: int = 1024,
            n_mel_channels: int = 100,
            target_sample_rate: int = 24000,
            mask_ratio_start: float = 0.7,
            mask_ratio_end: float = 1,
            project_out: bool = False,
            seq_len: int = 236,
    ):


        assert mel_spec_type in self.Mel_MODELS, f"Unknown TTS model name: {mel_spec_type}"
        super().__init__(self.MEL_MODEL_DIMS[mel_spec_type], output_dim, project_out=project_out)
        
        self.mel_spec_type = mel_spec_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        if mel_spec_type == "mel_features":
            self.extractor = get_vocos_mel_spectrogram

        self.proj_features = nn.Linear(in_features=self.n_mel_channels, out_features=768)     
        self.proj_sequence_features = nn.Linear(in_features=1723, out_features=seq_len)

        self.empty_audio_feat = nn.Parameter(torch.zeros(1, seq_len, 768), requires_grad=True)
        self.norm = RMSNorm(768)
        
    def resample_and_pad(self, wavs, device):
        # Resample audio tensors
        wavs = [wav.to(device).float() for wav in wavs]
        
        # Find the maximum length
        max_length = max([wav.shape[-1] for wav in wavs])
        
        # Pad audio tensors to the maximum length
        padded_wavs = [torch.nn.functional.pad(wav, (0, max_length - wav.shape[-1])) for wav in wavs]
        # Stack the padded audio tensors
        stacked_wavs = torch.stack(padded_wavs)
        # Resample the entire batch of audio tensors
        resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sample_rate).to(device)
        resampled_wavs = resampler(stacked_wavs)
        return resampled_wavs

    def mask_mel_spectrogram(self, mels: torch.Tensor, task_tts_list: tp.List[bool] = None) -> torch.Tensor:
        """
        Randomly mask the mel spectrogram
        Args:
            mels: tensor with shape [b, 100, t]
            task_tts_list: list indicating whether each sample is a TTS task; only True samples will be masked
        Returns:
            masked_mels: the masked mel spectrogram
        """
        batch_size, n_mels, seq_len = mels.shape
        device = mels.device
        
        # Check if the model is trainable
        model_trainable = any(p.requires_grad for p in self.parameters())
        if model_trainable:
            # Randomly generate mask ratio for each sample
            mask_ratios = torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        else:
            mask_ratios = torch.zeros(batch_size, device=device)
        
        # Generate mask for each sample
        masked_mels = mels.clone()
        for i in range(batch_size):
            # Only mask samples where TASK_TTS is True
            if task_tts_list[i]:
                # Calculate the length to be masked
                mask_len = int(seq_len * mask_ratios[i])
                # Randomly select the start position
                start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
                # Create mask
                masked_mels[i, :, start_pos:start_pos + mask_len] = 0
            
        return masked_mels, mask_ratios

    def forward(self, audio_input_prompts: tp.List[dict], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        wavs = []
        TASK_TTS = []
        for audio_input_prompt in audio_input_prompts:
            wavs.append(audio_input_prompt['audio_input_wav'])
            TASK_TTS.append(audio_input_prompt['TASK_TTS'])
        
        self.proj_out.to(device)
        wavs = torch.stack(wavs, dim=0).to(device).float()

        mels = self.extractor(
            waveform=wavs,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        # Mask the mel spectrogram, only for samples where TASK_TTS is True
        if self.mask_ratio_start < self.mask_ratio_end:
            mels, mask_ratios = self.mask_mel_spectrogram(mels, TASK_TTS)

        mels = self.proj_sequence_features(mels)
        mels = mels.transpose(1, 2)
        mels_emb = self.proj_features(mels)

        mels_emb = self.norm(mels_emb)

        return mels_emb, torch.ones(mels_emb.shape[0], mels_emb.shape[1]).to(device) 
    

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)
            
            output[key] = conditioner(conditioner_inputs, device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "qwen_omni":
            conditioners[id] = OmniConditioner(**conditioner_config)
        elif conditioner_type == "tts-f5" or conditioner_type == "tts-f5-token-level":
            conditioners[id] = TTSConditioner(**conditioner_config)             
        elif conditioner_type == "sync_feature" or conditioner_type == "video_sync_feature":
            conditioners[id] = SynchformerConditioner(**conditioner_config)
        elif conditioner_type == "mel_spec" or conditioner_type == "mel_spec-w-empty-feature":
            conditioners[id] = AudioMelConditioner(**conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)