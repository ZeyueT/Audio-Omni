"""
Latent Forcing Diffusion Transformer for Audio Generation.

Faithfully adapted from the Latent Forcing paper (arXiv:2602.11401):

Original (image):
  pixel_embed(x_pixel) + dino_embed(x_dino) → shared Transformer → depth-wise heads

Ours (audio):
  audio_patch_embed(raw_waveform) + latent_proj(vae_latent) → shared Transformer → depth-wise heads

Key design choices (matching the paper exactly):
  1. Two paths are ADDED element-wise (not concatenated) → same seq length
  2. ALL shared Transformer layers process the summed representation
  3. Depth-wise output heads are INDEPENDENT Transformer blocks (JiTBlock-style)
     with their own hidden_size, followed by FinalLayer
  4. adaLN conditioning: c = t_audio_emb + t_latent_emb + global_cond
  5. Returns tuple (output_audio, output_latent)

Pretrained weight compatibility:
  The shared Transformer blocks use the SAME attribute naming as the original
  DiffusionTransformer (timestep_features, to_timestep_embed, to_cond_embed,
  transformer.layers, transformer.rotary_pos_emb, etc.) so that pretrained
  checkpoints can be loaded directly via copy_state_dict without key remapping.

This file does NOT modify any existing class.
"""

import typing as tp
import math

import torch
import torch.utils.checkpoint as torch_ckpt
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .blocks import FourierFeatures
from .transformer import TransformerBlock, RotaryEmbedding


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_ckpt.checkpoint(function, *args, **kwargs)


# --------------------------------------------------------------------------- #
#  adaLN-based Transformer Block (JiTBlock style, for depth-wise heads)        #
# --------------------------------------------------------------------------- #

class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with adaLN modulation, matching JiTBlock from the paper.

    adaLN: x = x * (1 + scale) + shift, gated residual
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 cond_dim: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        # Pre-norm
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)

        # FFN (SwiGLU)
        ffn_hidden = int(hidden_size * mlp_ratio * 2 / 3)
        self.ffn_w12 = nn.Linear(hidden_size, 2 * ffn_hidden, bias=True)
        self.ffn_w3 = nn.Linear(ffn_hidden, hidden_size, bias=True)

        # adaLN modulation: 6 * hidden_size
        cond_dim = cond_dim or hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, rotary_pos_emb=None):
        """
        Args:
            x: (B, N, D) input tokens
            c: (B, D_c) conditioning vector (adaLN)
            rotary_pos_emb: optional RoPE (freqs, scale) from RotaryEmbedding
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with adaLN
        B, N, D = x.shape
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if available
        if rotary_pos_emb is not None:
            from .transformer import apply_rotary_pos_emb as _apply_rope
            freqs, scale = rotary_pos_emb
            q = _apply_rope(q, freqs, scale)
            k = _apply_rope(k, freqs, scale)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.attn_proj(attn_out)

        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN with adaLN
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x12 = self.ffn_w12(h)
        x1, x2 = x12.chunk(2, dim=-1)
        ffn_out = self.ffn_w3(F.silu(x1) * x2)

        x = x + gate_mlp.unsqueeze(1) * ffn_out

        return x


# --------------------------------------------------------------------------- #
#  FinalLayer (adaLN + Linear, matching the paper)                             #
# --------------------------------------------------------------------------- #

class FinalLayer(nn.Module):
    """Final output layer with adaLN modulation."""

    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int = None):
        super().__init__()
        cond_dim = cond_dim or hidden_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


# --------------------------------------------------------------------------- #
#  1-D Patch Embedding for raw audio waveform                                  #
# --------------------------------------------------------------------------- #

class AudioPatchEmbed(nn.Module):
    """
    Converts raw audio waveform into patch tokens.

    Input:  (B, audio_channels, T_samples)   e.g. (B, 2, 485100)
    Output: (B, T_patches, embed_dim)         e.g. (B, 237, 2048)
    """

    def __init__(self, audio_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            audio_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        remainder = T % self.patch_size
        if remainder != 0:
            x = F.pad(x, (0, self.patch_size - remainder))
        x = self.proj(x)          # (B, embed_dim, T_patches)
        x = x.transpose(1, 2)     # (B, T_patches, embed_dim)
        return x


class AudioPatchUnembed(nn.Module):
    """
    Converts patch tokens back to raw audio waveform.

    Input:  (B, T_patches, out_channels_audio)
    Output: (B, audio_channels, T_samples)
    """

    def __init__(self, out_channels_audio: int, audio_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.audio_channels = audio_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_patches, audio_channels * patch_size)
        → (B, audio_channels, T_patches * patch_size)
        """
        B, T, _ = x.shape
        x = x.reshape(B, T, self.audio_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (B, C, T, P)
        x = x.reshape(B, self.audio_channels, T * self.patch_size)
        return x


# --------------------------------------------------------------------------- #
#  Latent Embedding (VAE latent → embed_dim)                                   #
# --------------------------------------------------------------------------- #

class LatentEmbed(nn.Module):
    """
    Projects VAE latent tokens to embed_dim.

    Input:  (B, latent_channels, T_latent)  e.g. (B, 64, 237)
    Output: (B, T_latent, embed_dim)
    """

    def __init__(self, latent_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_channels, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)     # (B, T_latent, latent_channels)
        x = self.proj(x)          # (B, T_latent, embed_dim)
        return x


# --------------------------------------------------------------------------- #
#  SharedTransformer — wraps layers + rotary_pos_emb for key compatibility     #
# --------------------------------------------------------------------------- #

class SharedTransformer(nn.Module):
    """
    A thin container that holds:
      - layers: nn.ModuleList of TransformerBlock
      - rotary_pos_emb: RotaryEmbedding

    This exists purely so that the state_dict keys are:
      transformer.layers.{i}.xxx
      transformer.rotary_pos_emb.xxx
    matching the original ContinuousTransformer naming.
    """

    def __init__(self, layers: nn.ModuleList, rotary_pos_emb: RotaryEmbedding):
        super().__init__()
        self.layers = layers
        self.rotary_pos_emb = rotary_pos_emb


# --------------------------------------------------------------------------- #
#  LatentForcingDiffusionTransformer                                           #
# --------------------------------------------------------------------------- #

class LatentForcingDiffusionTransformer(nn.Module):
    """
    Faithful adaptation of JiTCoT for audio.

    Architecture (matching the paper):
      1. audio_patch_embed(raw_waveform) + latent_embed(vae_latent) → x
      2. Shared Transformer blocks (all `depth` layers, adaLN with c)
      3. Depth-wise heads: independent Transformer blocks per path
      4. FinalLayer per path → output_audio, output_latent

    Pretrained weight compatibility:
      Shared parts use the SAME attribute names as DiffusionTransformer:
        - timestep_features, to_timestep_embed  (audio timestep)
        - to_cond_embed, to_global_embed, to_prepend_embed
        - transformer.layers.{i}, transformer.rotary_pos_emb
      so that copy_state_dict can load pretrained DiT weights directly.
    """

    def __init__(
        self,
        # --- audio path ---
        audio_channels: int = 2,
        audio_patch_size: int = 2048,
        # --- latent path ---
        latent_channels: int = 64,
        # --- transformer ---
        embed_dim: int = 2048,
        depth: int = 36,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        # --- depth-wise heads ---
        dh_depth: int = 2,
        dh_hidden_size: int = 1024,
        dh_num_heads: int = 16,
        # --- conditioning ---
        cond_token_dim: int = 768,
        project_cond_tokens: bool = True,
        global_cond_dim: int = 768,
        project_global_cond: bool = True,
        prepend_cond_dim: int = 0,
        input_concat_dim: int = 0,
        # --- misc ---
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        **kwargs,
    ):
        super().__init__()

        self.audio_channels = audio_channels
        self.audio_patch_size = audio_patch_size
        self.latent_channels = latent_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.dh_depth = dh_depth
        self.patch_size = 1  # for min_input_length compatibility
        self.global_cond_type = global_cond_type
        self.cond_token_dim = cond_token_dim

        # Output channels
        self.out_channels_audio = audio_channels * audio_patch_size  # for unpatchify
        self.out_channels_latent = latent_channels

        # ================================================================== #
        # ================================================================== #
        timestep_features_dim = 256

        # Audio timestep (matches DiffusionTransformer exactly)
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        # Latent timestep (new, won't conflict)
        self.t_latent_features = FourierFeatures(1, timestep_features_dim)
        self.to_t_latent_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        # ================================================================== #
        #  Conditioning projections (SAME names as DiffusionTransformer)      #
        # ================================================================== #
        if cond_token_dim > 0:
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        # ================================================================== #
        #  Input embeddings: element-wise ADD (like the paper)                #
        # ================================================================== #
        self.audio_embedder = AudioPatchEmbed(audio_channels, embed_dim, audio_patch_size)
        self.latent_embedder = LatentEmbed(latent_channels, embed_dim)

        # ================================================================== #
        # ================================================================== #
        dim_heads = embed_dim // num_heads

        _ignore_keys = {
            'condition_mask_type', 'video_fps', 'input_concat_dim',
            'prepend_cond_dim', 'audio_channels', 'audio_patch_size',
            'latent_channels', 'dh_depth', 'dh_hidden_size', 'dh_num_heads',
            'split_depth', 'mlp_ratio',
        }
        block_kwargs = {k: v for k, v in kwargs.items() if k not in _ignore_keys}

        layers = nn.ModuleList()
        for i in range(depth):
            layers.append(
                TransformerBlock(
                    embed_dim,
                    dim_heads=dim_heads,
                    cross_attend=cond_token_dim > 0,
                    dim_context=cond_embed_dim if cond_token_dim > 0 else None,
                    global_cond_dim=embed_dim,  # adaLN: c vector
                    zero_init_branch_outputs=True,
                    layer_ix=i,
                    **block_kwargs,
                )
            )

        rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))

        # Wrap in SharedTransformer for key naming compatibility
        self.transformer = SharedTransformer(layers, rotary_pos_emb)

        # ================================================================== #
        #  Depth-wise heads (independent Transformer blocks per path)         #
        # ================================================================== #
        if dh_depth > 0:
            # Project from embed_dim to dh_hidden_size
            self.dh_audio_proj = nn.Linear(embed_dim, dh_hidden_size)
            self.dh_latent_proj = nn.Linear(embed_dim, dh_hidden_size)

            dh_num_heads = dh_num_heads or (num_heads * dh_hidden_size // embed_dim)

            self.dh_blocks_audio = nn.ModuleList([
                AdaLNTransformerBlock(
                    dh_hidden_size, dh_num_heads,
                    mlp_ratio=mlp_ratio, cond_dim=embed_dim,
                ) for _ in range(dh_depth)
            ])
            self.dh_blocks_latent = nn.ModuleList([
                AdaLNTransformerBlock(
                    dh_hidden_size, dh_num_heads,
                    mlp_ratio=mlp_ratio, cond_dim=embed_dim,
                ) for _ in range(dh_depth)
            ])

            self.final_layer_audio = FinalLayer(
                dh_hidden_size, self.out_channels_audio, cond_dim=embed_dim,
            )
            self.final_layer_latent = FinalLayer(
                dh_hidden_size, self.out_channels_latent, cond_dim=embed_dim,
            )
        else:
            # Single final layer outputting both
            self.final_layer = FinalLayer(
                embed_dim, self.out_channels_audio + self.out_channels_latent,
            )

        # Audio unpatchify
        self.audio_unpatchify = AudioPatchUnembed(
            self.out_channels_audio, audio_channels, audio_patch_size,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Re-zero adaLN gates in shared blocks
        for block in self.transformer.layers:
            if hasattr(block, 'to_scale_shift_gate'):
                nn.init.zeros_(block.to_scale_shift_gate[-1].weight)

        # Re-zero depth-wise heads
        if self.dh_depth > 0:
            for block_list in [self.dh_blocks_audio, self.dh_blocks_latent]:
                for block in block_list:
                    nn.init.zeros_(block.adaLN_modulation[-1].weight)
                    nn.init.zeros_(block.adaLN_modulation[-1].bias)

            for fl in [self.final_layer_audio, self.final_layer_latent]:
                nn.init.zeros_(fl.adaLN_modulation[-1].weight)
                nn.init.zeros_(fl.adaLN_modulation[-1].bias)
                nn.init.zeros_(fl.linear.weight)
                nn.init.zeros_(fl.linear.bias)

    # ---------------------------------------------------------------------- #
    def _forward(
        self,
        x_audio: torch.Tensor,      # (B, audio_channels, T_samples)
        x_latent: torch.Tensor,      # (B, latent_channels, T_latent)
        t_audio: torch.Tensor,       # (B,)
        t_latent: torch.Tensor,      # (B,)
        mask: torch.Tensor = None,
        cross_attn_cond: torch.Tensor = None,
        cross_attn_cond_mask: torch.Tensor = None,
        input_concat_cond: torch.Tensor = None,
        global_embed: torch.Tensor = None,
        prepend_cond: torch.Tensor = None,
        prepend_cond_mask: torch.Tensor = None,
        return_info: bool = False,
        **kwargs,
    ):
        T_samples = x_audio.shape[2]
        T_latent_orig = x_latent.shape[2]

        info = {"hidden_states": []} if return_info else None

        # ==================== conditioning ==================== #
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        # Global conditioning → adaLN vector c
        # Paper: c = t_pixel_emb + t_dino_emb + y_emb
        t_emb_audio = self.to_timestep_embed(
            self.timestep_features(t_audio[:, None])
        )  # (B, embed_dim)
        t_emb_latent = self.to_t_latent_embed(
            self.t_latent_features(t_latent[:, None])
        )  # (B, embed_dim)
        c = t_emb_audio + t_emb_latent  # (B, embed_dim)

        if global_embed is not None:
            global_proj = self.to_global_embed(global_embed)  # (B, ?, embed_dim)
            if global_proj.ndim == 3:
                global_proj = global_proj.mean(dim=1)  # pool to (B, embed_dim)
            c = c + global_proj

        # ==================== prepend conditioning ==================== #
        prepend_inputs = None
        prepend_mask_out = None
        prepend_length = 0

        if prepend_cond is not None and hasattr(self, 'to_prepend_embed'):
            prepend_inputs = self.to_prepend_embed(prepend_cond)
            prepend_mask_out = prepend_cond_mask
            prepend_length = prepend_inputs.shape[1]

        # ==================== input embedding: element-wise ADD ==================== #
        x_a = self.audio_embedder(x_audio)    # (B, T_patches, embed_dim)
        x_l = self.latent_embedder(x_latent)  # (B, T_latent, embed_dim)

        # Align sequence lengths (may differ by 1 due to rounding in patch/VAE)
        T_a, T_l = x_a.shape[1], x_l.shape[1]
        if T_a > T_l:
            # Pad latent tokens with zeros at the end
            x_l = F.pad(x_l, (0, 0, 0, T_a - T_l))
        elif T_l > T_a:
            # Pad audio tokens with zeros at the end
            x_a = F.pad(x_a, (0, 0, 0, T_l - T_a))

        x = x_a + x_l  # Element-wise add (like the paper!)

        T_tokens = x.shape[1]

        # ==================== prepend tokens ==================== #
        if prepend_inputs is not None:
            x = torch.cat([prepend_inputs, x], dim=1)

        # ==================== mask ==================== #
        if mask is not None:
            if prepend_inputs is not None:
                prepend_ones = prepend_mask_out if prepend_mask_out is not None else \
                    torch.ones((mask.shape[0], prepend_length), device=mask.device, dtype=torch.bool)
                mask = torch.cat([prepend_ones, mask], dim=1)

        # ==================== RoPE ==================== #
        rotary_pos_emb = self.transformer.rotary_pos_emb.forward_from_seq_len(x.shape[1])

        # ==================== shared Transformer blocks ==================== #
        for block in self.transformer.layers:
            x = checkpoint(
                block, x,
                context=cross_attn_cond,
                context_mask=cross_attn_cond_mask,
                mask=mask,
                rotary_pos_emb=rotary_pos_emb,
                global_cond=c,  # adaLN conditioning
            )
            if return_info:
                info["hidden_states"].append(x)

        # ==================== strip prepend tokens ==================== #
        if prepend_length > 0:
            x = x[:, prepend_length:, :]

        # ==================== depth-wise heads ==================== #
        # RoPE for depth-wise heads (seq length = T_tokens)
        rope_dh = self.transformer.rotary_pos_emb.forward_from_seq_len(T_tokens)

        if self.dh_depth > 0:
            # Project to dh_hidden_size
            dh_audio = self.dh_audio_proj(x)    # (B, T_tokens, dh_hidden_size)
            dh_latent = self.dh_latent_proj(x)  # (B, T_tokens, dh_hidden_size)

            # Independent Transformer blocks
            for block in self.dh_blocks_audio:
                dh_audio = checkpoint(block, dh_audio, c)
            for block in self.dh_blocks_latent:
                dh_latent = checkpoint(block, dh_latent, c)

            # Final layers
            out_audio_raw = self.final_layer_audio(dh_audio, c)
            out_latent_raw = self.final_layer_latent(dh_latent, c)
        else:
            combined = self.final_layer(x, c)
            out_audio_raw, out_latent_raw = combined.split(
                [self.out_channels_audio, self.out_channels_latent], dim=-1
            )

        # ==================== unpatchify audio ==================== #
        out_audio = self.audio_unpatchify(out_audio_raw)
        out_audio = out_audio[:, :, :T_samples]

        # ==================== latent output ==================== #
        out_latent = out_latent_raw.transpose(1, 2)  # (B, latent_channels, T_tokens)
        out_latent = out_latent[:, :, :T_latent_orig]  # Truncate to original latent length

        if return_info:
            return (out_audio, out_latent), info
        return out_audio, out_latent


# --------------------------------------------------------------------------- #
#  LatentForcingDiTWrapper                                                     #
# --------------------------------------------------------------------------- #

class LatentForcingDiTWrapper(nn.Module):
    """
    Wrapper that plugs into ConditionedDiffusionModelWrapper.

    Accepts x as tuple (x_audio, x_latent), t as tuple (t_audio, t_latent).
    Handles CFG dropout and batch-CFG.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.supports_cross_attention = True
        self.supports_global_cond = False
        self.supports_input_concat = False
        self.supports_prepend_cond = True

        self.model = LatentForcingDiffusionTransformer(**kwargs)

        self.patch_size = 1
        self.io_channels = kwargs.get("latent_channels", 64)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        negative_input_concat_cond=None,
        global_cond=None,
        negative_global_cond=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        batch_cfg: bool = True,
        rescale_cfg: bool = False,
        scale_phi: float = 0.0,
        mask=None,
        return_info: bool = False,
        **kwargs,
    ):
        # --- unpack tuples ---
        if isinstance(x, (tuple, list)):
            x_audio, x_latent = x
        else:
            raise ValueError("LatentForcingDiTWrapper expects x as tuple (x_audio, x_latent)")

        if isinstance(t, (tuple, list)):
            t_audio, t_latent = t
        else:
            t_audio = t_latent = t

        # --- mask conversion ---
        if cross_attn_mask is not None:
            cross_attn_mask = cross_attn_mask.bool()
        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # --- CFG dropout (training) ---
        if cfg_dropout_prob > 0.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond)
                dropout_mask = torch.bernoulli(
                    torch.full((cross_attn_cond.shape[0], 1, 1),
                               cfg_dropout_prob, device=cross_attn_cond.device)
                ).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond)
                dropout_mask = torch.bernoulli(
                    torch.full((prepend_cond.shape[0], 1, 1),
                               cfg_dropout_prob, device=prepend_cond.device)
                ).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        # ============================================================== #
        #  Batch CFG (inference)                                          #
        # ============================================================== #
        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None):
            batch_x_audio = torch.cat([x_audio, x_audio], dim=0)
            batch_x_latent = torch.cat([x_latent, x_latent], dim=0)
            batch_t_audio = torch.cat([t_audio, t_audio], dim=0)
            batch_t_latent = torch.cat([t_latent, t_latent], dim=0)

            batch_global_cond = (
                torch.cat([global_cond, global_cond], dim=0)
                if global_cond is not None else None
            )
            batch_mask = (
                torch.cat([mask, mask], dim=0) if mask is not None else None
            )

            # cross-attention: cond / null
            batch_cond = None
            batch_cond_masks = None
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond)
                if negative_cross_attn_cond is not None:
                    if negative_cross_attn_mask is not None:
                        neg_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                        negative_cross_attn_cond = torch.where(
                            neg_mask, negative_cross_attn_cond, null_embed
                        )
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)
                if cross_attn_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_mask, cross_attn_mask], dim=0)

            # prepend cond
            batch_prepend = None
            batch_prepend_mask = None
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond)
                batch_prepend = torch.cat([prepend_cond, null_embed], dim=0)
                if prepend_cond_mask is not None:
                    batch_prepend_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)

            batch_output = self.model._forward(
                batch_x_audio, batch_x_latent, batch_t_audio, batch_t_latent,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend,
                prepend_cond_mask=batch_prepend_mask,
                mask=batch_mask,
                return_info=return_info,
                **kwargs,
            )

            if return_info:
                (batch_out_audio, batch_out_latent), batch_info = batch_output
            else:
                batch_out_audio, batch_out_latent = batch_output

            cond_a, uncond_a = batch_out_audio.chunk(2, dim=0)
            cond_l, uncond_l = batch_out_latent.chunk(2, dim=0)
            cfg_a = uncond_a + (cond_a - uncond_a) * cfg_scale
            cfg_l = uncond_l + (cond_l - uncond_l) * cfg_scale

            if scale_phi != 0.0:
                cond_std_a = cond_a.std(dim=1, keepdim=True)
                cfg_std_a = cfg_a.std(dim=1, keepdim=True)
                cfg_a = scale_phi * (cfg_a * (cond_std_a / cfg_std_a)) + (1 - scale_phi) * cfg_a
                cond_std_l = cond_l.std(dim=1, keepdim=True)
                cfg_std_l = cfg_l.std(dim=1, keepdim=True)
                cfg_l = scale_phi * (cfg_l * (cond_std_l / cfg_std_l)) + (1 - scale_phi) * cfg_l

            if return_info:
                return (cfg_a, cfg_l), batch_info
            return cfg_a, cfg_l

        # ============================================================== #
        #  Normal forward (no CFG)                                        #
        # ============================================================== #
        return self.model._forward(
            x_audio, x_latent, t_audio, t_latent,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            global_embed=global_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            mask=mask,
            return_info=return_info,
            **kwargs,
        )
