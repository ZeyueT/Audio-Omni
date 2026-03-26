import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Sparsemax激活函数实现
class Sparsemax(nn.Module):
    def forward(self, input):
        input = input - input.max(dim=-1, keepdim=True)[0]
        dim = -1
        zs = torch.sort(input, descending=True, dim=dim)[0]
        range = torch.arange(1, input.size(dim)+1, device=input.device, dtype=input.dtype).view(
            [1]*(input.dim()-1) + [input.size(dim)])
        bound = 1 + range * zs
        cumulative = torch.cumsum(zs, dim)
        is_gt = (bound > cumulative).type(input.dtype)
        k = torch.max(is_gt * range, dim=dim, keepdim=True)[0]
        tau = (torch.sum(is_gt * zs, dim=dim, keepdim=True) - 1) / k
        output = torch.clamp(input - tau, min=0)
        return output

# Cross-Attention模块
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    def forward(self, query, key, value):
        out, _ = self.attn(query, key, value)
        return out

# Gated XATTN-DENSE层
class DynamicSparseGatedXATTN_DENSE(nn.Module):
    def __init__(self, d_model, n_heads,
                 use_sparse_gating=True,
                 use_token_gate=True,
                 use_global_gate=True,
                 gating_on_attention=True,
                 equal_fusion=False):
        super().__init__()
        self.cross_attn_v = CrossAttention(d_model, n_heads)
        self.cross_attn_a = CrossAttention(d_model, n_heads)
        self.cross_attn_t = CrossAttention(d_model, n_heads)
        self.use_sparse_gating = use_sparse_gating
        self.use_token_gate = use_token_gate
        self.use_global_gate = use_global_gate
        self.gating_on_attention = gating_on_attention
        self.equal_fusion = equal_fusion

        if use_sparse_gating:
            self.gate_act = Sparsemax()
        else:
            self.gate_act = nn.Softmax(dim=-1)  # 或nn.Sigmoid()

        if use_token_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_model * 4, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 3)
            )
        else:
            self.global_gate_logits = nn.Parameter(torch.zeros(1, 1, 3))  # [1, 1, 3]

        if use_global_gate:
            self.global_gate = nn.Parameter(torch.tensor(0.5))

        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, main_tokens, vision_tokens, audio_tokens, text_tokens):
        attn_v = self.cross_attn_v(main_tokens, vision_tokens, vision_tokens)
        attn_a = self.cross_attn_a(main_tokens, audio_tokens, audio_tokens)
        attn_t = self.cross_attn_t(main_tokens, text_tokens, text_tokens)

        if self.equal_fusion:
            # 所有模态权重固定为1/3
            fused = (attn_v + attn_a + attn_t) / 3
        else:
            if self.use_token_gate:
                concat = torch.cat([main_tokens, attn_v, attn_a, attn_t], dim=-1)
                gates = self.gate_mlp(concat)  # [batch, seq, 3]
                gates = self.gate_act(gates)
            else:
                gates = self.gate_act(self.global_gate_logits.expand(main_tokens.size(0), main_tokens.size(1), 3))

            gate_v = gates[..., 0:1]
            gate_a = gates[..., 1:2]
            gate_t = gates[..., 2:3]

            if self.gating_on_attention:
                fused = gate_v * attn_v + gate_a * attn_a + gate_t * attn_t
            else:
                # 门控后处理（如拼接后MLP）
                fused = torch.cat([gate_v * attn_v, gate_a * attn_a, gate_t * attn_t], dim=-1)
                fused = nn.Linear(fused.size(-1), main_tokens.size(-1)).to(fused.device)(fused)

        if self.use_global_gate:
            fused = self.global_gate * fused + (1 - self.global_gate) * main_tokens
        else:
            fused = fused + main_tokens

        fused = self.norm1(fused)
        dense_out = self.mlp(fused)
        out = self.norm2(fused + dense_out)
        return out, gates


class MLF_Block(nn.Module):
    """
    实现了“统一评分，结构化分裂”的融合机制。
    *** 此版本假设video, text, audio三个模态总是同时作为输入。***

    工作流程:
    1. 宏观门控: 对V, T, A输入进行初步的加权筛选。
    2. 统一评分: 创建一支统一的专家团队，通过Cross-Attention与所有
       拼接后的模态信息(gated_V+T+A)交互。
    3. 全局融合: 更新后的专家团队在一个共享的Transformer中进行深度融合。
    4. 结构化分裂与精炼: 融合后的专家团队被拆分为三组，生成针对
       V, T, A的精炼信号，并通过旁路门控以残差形式添加到原始输入上。
    """
    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 2
        
        # 1. 宏观门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 2), nn.Sigmoid()
        )

        # 2. 统一专家团队
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        # 3. 统一评分与融合组件
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 4. 分裂与精炼组件
        self.norm_highlevel_tokens = nn.LayerNorm(dim)
        self.norm_lowlevel_tokens = nn.LayerNorm(dim)

        self.bypass_gate_highlevel_tokens = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_lowlevel_tokens = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 初始化Cross-Attention的输出投影为0，有助于稳定训练初期
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
        self,
        highlevel_tokens: torch.Tensor,
        lowlevel_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = highlevel_tokens.shape[0]

        highlevel_tokens_global = highlevel_tokens.mean(dim=1)
        lowlevel_tokens_global = lowlevel_tokens.mean(dim=1)
        
        all_global = torch.cat([highlevel_tokens_global, lowlevel_tokens_global], dim=1)

        gates = self.gating_network(all_global)
        w_highlevel_tokens, w_lowlevel_tokens = gates.chunk(2, dim=-1)

        gated_highlevel_tokens = highlevel_tokens * w_highlevel_tokens.unsqueeze(-1)
        gated_lowlevel_tokens = lowlevel_tokens * w_lowlevel_tokens.unsqueeze(-1)

        full_context = torch.cat([gated_highlevel_tokens, gated_lowlevel_tokens], dim=1)

        # --- 2. 统一评分 ---
        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)
        
        # --- 3. 全局融合 ---
        fused_experts = self.fusion_transformer(updated_experts)

        # --- 4. 结构化分裂与精炼 ---
        fused_highlevel_experts, fused_lowlevel_experts = fused_experts.chunk(2, dim=1)
        
        # 计算每个模态的精炼信号
        refinement_highlevel_tokens = fused_highlevel_experts.mean(dim=1)
        refinement_lowlevel_tokens = fused_lowlevel_experts.mean(dim=1)

        # 获取旁路门控权重
        alpha_highlevel_tokens = torch.sigmoid(self.bypass_gate_highlevel_tokens)
        alpha_lowlevel_tokens = torch.sigmoid(self.bypass_gate_lowlevel_tokens)

        # 与原始输入进行残差连接
        final_highlevel_tokens = highlevel_tokens + alpha_highlevel_tokens * self.norm_highlevel_tokens(refinement_highlevel_tokens).unsqueeze(1)
        final_lowlevel_tokens = lowlevel_tokens + alpha_lowlevel_tokens * self.norm_lowlevel_tokens(refinement_lowlevel_tokens).unsqueeze(1)

        return {
            "highlevel_tokens": final_highlevel_tokens,
            "lowlevel_tokens": final_lowlevel_tokens
        }


class MALF_Block(nn.Module):
    """
    实现了“统一评分，结构化分裂”的融合机制。
    *** 此版本假设video, text, audio三个模态总是同时作为输入。***

    工作流程:
    1. 宏观门控: 对V, T, A输入进行初步的加权筛选。
    2. 统一评分: 创建一支统一的专家团队，通过Cross-Attention与所有
       拼接后的模态信息(gated_V+T+A)交互。
    3. 全局融合: 更新后的专家团队在一个共享的Transformer中进行深度融合。
    4. 结构化分裂与精炼: 融合后的专家团队被拆分为三组，生成针对
       V, T, A的精炼信号，并通过旁路门控以残差形式添加到原始输入上。
    """
    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 2
        
        # 1. 宏观门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 2), nn.Sigmoid()
        )

        # 2. 统一专家团队
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        # 3. 统一评分与融合组件
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 4. 分裂与精炼组件
        self.norm_m1_tokens = nn.LayerNorm(dim)
        self.norm_m2_tokens = nn.LayerNorm(dim)

        self.bypass_gate_m1_tokens = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_m2_tokens = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 初始化Cross-Attention的输出投影为0，有助于稳定训练初期
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
        self,
        m1_tokens: torch.Tensor,
        m2_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = m1_tokens.shape[0]

        m1_tokens_global = m1_tokens.mean(dim=1)
        m2_tokens_global = m2_tokens.mean(dim=1)
        
        all_global = torch.cat([m1_tokens_global, m2_tokens_global], dim=1)

        gates = self.gating_network(all_global)
        w_m1_tokens, w_m2_tokens = gates.chunk(2, dim=-1)

        gated_m1_tokens = m1_tokens * w_m1_tokens.unsqueeze(-1)
        gated_m2_tokens = m2_tokens * w_m2_tokens.unsqueeze(-1)

        full_context = torch.cat([gated_m1_tokens, gated_m2_tokens], dim=1)

        # --- 2. 统一评分 ---
        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)
        
        # --- 3. 全局融合 ---
        fused_experts = self.fusion_transformer(updated_experts)

        # --- 4. 结构化分裂与精炼 ---
        fused_m1_experts, fused_m2_experts = fused_experts.chunk(2, dim=1)
        
        # 计算每个模态的精炼信号
        refinement_m1_tokens = fused_m1_experts.mean(dim=1)
        refinement_m2_tokens = fused_m2_experts.mean(dim=1)

        # 获取旁路门控权重
        alpha_m1_tokens = torch.sigmoid(self.bypass_gate_m1_tokens)
        alpha_m2_tokens = torch.sigmoid(self.bypass_gate_m2_tokens)

        # 与原始输入进行残差连接
        final_m1_tokens = m1_tokens + alpha_m1_tokens * self.norm_m1_tokens(refinement_m1_tokens).unsqueeze(1)
        final_m2_tokens = m2_tokens + alpha_m2_tokens * self.norm_m2_tokens(refinement_m2_tokens).unsqueeze(1)

        return {
            "m1_tokens": final_m1_tokens,
            "m2_tokens": final_m2_tokens
        }


class MAF_Block(nn.Module):
    """
    实现了“统一评分，结构化分裂”的融合机制。
    *** 此版本假设video, text, audio三个模态总是同时作为输入。***

    工作流程:
    1. 宏观门控: 对V, T, A输入进行初步的加权筛选。
    2. 统一评分: 创建一支统一的专家团队，通过Cross-Attention与所有
       拼接后的模态信息(gated_V+T+A)交互。
    3. 全局融合: 更新后的专家团队在一个共享的Transformer中进行深度融合。
    4. 结构化分裂与精炼: 融合后的专家团队被拆分为三组，生成针对
       V, T, A的精炼信号，并通过旁路门控以残差形式添加到原始输入上。
    """
    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3
        
        # 1. 宏观门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )

        # 2. 统一专家团队
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        # 3. 统一评分与融合组件

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 4. 分裂与精炼组件
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 初始化Cross-Attention的输出投影为0，有助于稳定训练初期
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = video_tokens.shape[0]

        # --- 1. 宏观门控 ---
        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)
        
        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)

        # --- 2. 统一评分 ---
        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)
        
        # --- 3. 全局融合 ---
        fused_experts = self.fusion_transformer(updated_experts)

        # --- 4. 结构化分裂与精炼 ---
        fused_v_experts, fused_t_experts, fused_a_experts = fused_experts.chunk(3, dim=1)
        
        # 计算每个模态的精炼信号
        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        # 获取旁路门控权重
        alpha_v = torch.sigmoid(self.bypass_gate_v)
        alpha_t = torch.sigmoid(self.bypass_gate_t)
        alpha_a = torch.sigmoid(self.bypass_gate_a)

        # 与原始输入进行残差连接
        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a
        }


class MAF_Block_no_expert_update(nn.Module):
    """
    实现了“统一评分，结构化分裂”的融合机制。
    *** 此版本假设video, text, audio三个模态总是同时作为输入。***

    工作流程:
    1. 宏观门控: 对V, T, A输入进行初步的加权筛选。
    2. 统一评分: 创建一支统一的专家团队，通过Cross-Attention与所有
       拼接后的模态信息(gated_V+T+A)交互。
    3. 全局融合: 更新后的专家团队在一个共享的Transformer中进行深度融合。
    4. 结构化分裂与精炼: 融合后的专家团队被拆分为三组，生成针对
       V, T, A的精炼信号，并通过旁路门控以残差形式添加到原始输入上。
    """
    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3
        
        # 1. 宏观门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )

        # 2. 统一专家团队
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        # 3. 统一评分与融合组件

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 4. 分裂与精炼组件
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 初始化Cross-Attention的输出投影为0，有助于稳定训练初期
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = video_tokens.shape[0]

        # --- 1. 宏观门控 ---
        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)
        
        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)

        # --- 2. 统一评分 ---
        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        # updated_experts = self.norm1(experts + info)
        updated_experts = self.norm1(info)
        
        # --- 3. 全局融合 ---
        fused_experts = self.fusion_transformer(updated_experts)

        # --- 4. 结构化分裂与精炼 ---
        fused_v_experts, fused_t_experts, fused_a_experts = fused_experts.chunk(3, dim=1)
        
        # 计算每个模态的精炼信号
        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        # 获取旁路门控权重
        alpha_v = torch.sigmoid(self.bypass_gate_v)
        alpha_t = torch.sigmoid(self.bypass_gate_t)
        alpha_a = torch.sigmoid(self.bypass_gate_a)

        # 与原始输入进行残差连接
        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a
        }


class MAF_Block_wo_Gating(nn.Module):
    """
    消融实验 A1: 移除宏观门控网络 (Gating Network)。
    
    直接使用原始的输入tokens进行后续的专家评分和融合，
    不再进行模态级别的加权。
    """
    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        # --- 核心区别: 移除了 self.gating_network ---
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3
        
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(self, video_tokens: torch.Tensor, text_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = video_tokens.shape[0]

        # --- 核心区别: 直接拼接原始tokens，不再门控 ---
        full_context = torch.cat([video_tokens, text_tokens, audio_tokens], dim=1)

        # --- 后续流程与主模型相同 ---
        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)
        
        fused_experts = self.fusion_transformer(updated_experts)

        fused_v_experts, fused_t_experts, fused_a_experts = fused_experts.chunk(3, dim=1)
        
        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        alpha_v, alpha_t, alpha_a = torch.sigmoid(self.bypass_gate_v), torch.sigmoid(self.bypass_gate_t), torch.sigmoid(self.bypass_gate_a)

        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {"video": final_v, "text": final_t, "audio": final_a}


class MAF_Block_wo_Query(nn.Module):
    """
    消融实验 A2: 移除基于Query(Expert)的融合系统。
    
    使用一个更标准的融合方法：直接将门控后的tokens拼接，
    然后通过一个Transformer Encoder进行融合。
    """
    def __init__(self, dim: int, num_fusion_layers: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        
        # 1. 门控网络 (保留)
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        
        # --- 核心区别: 移除 experts 和 cross_attn ---
        # 替换为一个标准的 Transformer Encoder 用于直接融合
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 3. 精炼和旁路组件 (保留)
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))

    def forward(self, video_tokens: torch.Tensor, text_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = video_tokens.shape[0]

        # 1. 宏观门控 (保留)
        v_global = video_tokens.mean(dim=1); t_global = text_tokens.mean(dim=1); a_global = audio_tokens.mean(dim=1)
        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        # --- 核心区别: 直接融合拼接后的tokens ---
        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)
        fused_context = self.fusion_transformer(full_context)
        
        # 记录每个分支的长度，以便分裂
        len_v, len_t, len_a = video_tokens.shape[1], text_tokens.shape[1], audio_tokens.shape[1]
        
        # 分裂融合后的上下文
        fused_v = fused_context[:, :len_v, :]
        fused_t = fused_context[:, len_v:len_v+len_t, :]
        fused_a = fused_context[:, len_v+len_t:, :]
        
        # 计算精炼信号 (这里我们直接用融合后的结果作为精炼版)
        refinement_v = fused_v.mean(dim=1)
        refinement_t = fused_t.mean(dim=1)
        refinement_a = fused_a.mean(dim=1)

        alpha_v, alpha_t, alpha_a = torch.sigmoid(self.bypass_gate_v), torch.sigmoid(self.bypass_gate_t), torch.sigmoid(self.bypass_gate_a)

        # 与原始输入进行残差连接
        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {"video": final_v, "text": final_t, "audio": final_a}
    


class MAF_Block_wo_Alpha(nn.Module):
    """
    消融实验 A3 (最终版): 彻底移除残差连接。
    
    模块的输出直接是经过专家系统处理后生成的“精炼信号”本身，
    它将完全替换掉原始的输入tokens，而不是作为增量添加上去。
    这测试了融合模块独立生成表征的能力。
    """

    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3
        
        # 模块内部结构与主模型完全相同
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        
        # --- 核心区别: 移除了所有 bypass_gate 参数 ---

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = video_tokens.shape[0]

        # 1, 2, 3 步与主模型完全相同
        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)
        
        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)

        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)
        
        fused_experts = self.fusion_transformer(updated_experts)

        fused_v_experts, fused_t_experts, fused_a_experts = fused_experts.chunk(3, dim=1)
        
        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        # --- 核心区别: 直接返回处理后的信息，但需要注意维度 ---
        # 我们的精炼信号是 [B, D]，而原始输入是 [B, N, D]
        # 直接替换是不可能的。我们需要将精炼信号“广播”到原始的序列长度。
        
        # 1. 对精炼信号进行归一化
        norm_refinement_v = self.norm_v2(refinement_v)
        norm_refinement_t = self.norm_t2(refinement_t)
        norm_refinement_a = self.norm_a2(refinement_a)
        
        # 2. 扩展维度以匹配原始序列长度
        # [B, D] -> [B, 1, D] -> [B, N, D]
        final_v = norm_refinement_v.unsqueeze(1).expand_as(video_tokens)
        final_t = norm_refinement_t.unsqueeze(1).expand_as(text_tokens)
        final_a = norm_refinement_a.unsqueeze(1).expand_as(audio_tokens)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a
        }


class MAF_Block_GateAfterQuery(nn.Module):
    """
    消融实验变体：将宏观门控（Gating）作用于Query融合之后。

    工作流程:
    1. 统一评分: 专家团队直接与拼接后的原始V+T+A信息交互。
    2. 全局融合: 更新后的专家团队进行深度融合。
    3. 结构化分裂: 融合后的专家被拆分，生成“初步的”精炼信号。
    4. 后置门控与精炼: 宏观门控网络计算出的权重w_v, w_t, w_a
       被用来缩放这三个初步的精炼信号，然后通过旁路门控以
       残差形式添加到原始输入上。
    """
    def __init__(self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3
        
        # 1. 宏观门控网络 (保留，但作用位置改变)
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )

        # 2. 统一专家团队
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        # 3. 统一评分与融合组件
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 4. 分裂与精炼组件
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = video_tokens.shape[0]

        # --- 1. 计算门控权重 (但不立即使用) ---
        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)
        
        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        # --- 2. 统一评分 (直接在原始tokens上进行) ---
        # 核心区别: full_context 不再是 gated_tokens 的拼接
        full_context = torch.cat([video_tokens, text_tokens, audio_tokens], dim=1)

        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)
        
        # --- 3. 全局融合 ---
        fused_experts = self.fusion_transformer(updated_experts)

        # --- 4. 结构化分裂与后置门控精炼 ---
        fused_v_experts, fused_t_experts, fused_a_experts = fused_experts.chunk(3, dim=1)
        
        # 计算初步的精炼信号
        raw_refinement_v = fused_v_experts.mean(dim=1)
        raw_refinement_t = fused_t_experts.mean(dim=1)
        raw_refinement_a = fused_a_experts.mean(dim=1)

        # 核心区别: 在这里应用门控权重
        gated_refinement_v = raw_refinement_v * w_v
        gated_refinement_t = raw_refinement_t * w_t
        gated_refinement_a = raw_refinement_a * w_a
        
        # 获取旁路门控权重
        alpha_v = torch.sigmoid(self.bypass_gate_v)
        alpha_t = torch.sigmoid(self.bypass_gate_t)
        alpha_a = torch.sigmoid(self.bypass_gate_a)

        # 与原始输入进行残差连接
        final_v = video_tokens + alpha_v * self.norm_v2(gated_refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(gated_refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(gated_refinement_a).unsqueeze(1)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a
        }


class SplitAndFuseGatedRouter(nn.Module):
    """
    (完整文档同上)
    最终版本特性:
    - 引入可学习的旁路门控(Learnable Bypass Gate)，实现完美的“零影响启动”。
      模块在训练初期其输出严格等于输入，然后按需、平滑地学习引入复杂变换。
    """
    def __init__(self, dim: int, num_summary_queries: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim

        # --- 子模块定义 (与之前版本相同) ---
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        self.summary_queries = nn.Parameter(torch.randn(num_summary_queries, dim))
        self.gather_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gather_norm1 = nn.LayerNorm(dim)
        self.gather_self_attn_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.to_mu = nn.Linear(dim, dim)
        self.to_logvar = nn.Linear(dim, dim)
        self.scatter_v_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_t_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_a_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        
        # --- 核心改进：可学习的旁路门控参数 ---
        # 初始化为一个大负数，使其Sigmoid值接近0
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        # (可选但推荐) 仍然保留之前的初始化，让模块内部从一个简单的状态开始学习
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """初始化内部权重，使其从一个简单、稳定的状态开始学习。"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        
        # 初始化分发阶段注意力模块的输出投影为0
        scatter_attns = [
            getattr(m, 'scatter_v_attn', None),
            getattr(m, 'scatter_t_attn', None),
            getattr(m, 'scatter_a_attn', None)
        ]
        for attn in scatter_attns:
            if attn is not None and hasattr(attn, 'out_proj'):
                torch.nn.init.zeros_(attn.out_proj.weight)
                if attn.out_proj.bias is not None:
                    torch.nn.init.zeros_(attn.out_proj.bias)

    def _get_global_feature(self, tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tokens is None: return None
        return tokens.mean(dim=1)

    def forward(
        self,
        video_tokens: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        device = next(self.parameters()).device
        batch_size = -1
        for t in [video_tokens, text_tokens, audio_tokens]:
            if t is not None: batch_size = t.shape[0]; break
        if batch_size == -1: raise ValueError("至少需要提供一个模态作为输入。")

        # ====================================================================
        # === 步骤 1: 运行内部的融合精炼逻辑 (与之前几乎相同) ==============
        # ====================================================================
        v_global = self._get_global_feature(video_tokens)
        t_global = self._get_global_feature(text_tokens)
        a_global = self._get_global_feature(audio_tokens)

        if v_global is None: v_global = torch.zeros(batch_size, self.dim, device=device)
        if t_global is None: t_global = torch.zeros(batch_size, self.dim, device=device)
        if a_global is None: a_global = torch.zeros(batch_size, self.dim, device=device)
        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1) if video_tokens is not None else None
        gated_t = text_tokens * w_t.unsqueeze(-1) if text_tokens is not None else None
        gated_a = audio_tokens * w_a.unsqueeze(-1) if audio_tokens is not None else None
        context_list = [t for t in [gated_v, gated_t, gated_a] if t is not None]
        if not context_list:
            # 如果所有输入都为None，或者门控权重都为0，则直接返回原始输入
            return {k: v for k, v in [("video", video_tokens), ("text", text_tokens), ("audio", audio_tokens)] if v is not None}, None, None, {}

        gated_context = torch.cat(context_list, dim=1)

        # 确认这里是否和transformer block差不多
        queries = self.summary_queries.unsqueeze(0).expand(batch_size, -1, -1)
        extracted_info, _ = self.gather_cross_attn(queries, gated_context, gated_context) # attention里如果有residual，后面就不加了
        summary = self.gather_norm1(queries + extracted_info)
        fused_summary = self.gather_self_attn_block(summary)

        mu = self.to_mu(fused_summary)
        logvar = self.to_logvar(fused_summary)

        # "内部"的精炼输出
        # 考虑改成query*3而不是3个branch
        internal_refined_outputs = {}
        if gated_v is not None:
            v_refinement, _ = self.scatter_v_attn(query=gated_v, key=fused_summary, value=fused_summary)
            internal_refined_outputs["video"] = self.norm_v(gated_v + v_refinement)
        
        if gated_t is not None:
            t_refinement, _ = self.scatter_t_attn(query=gated_t, key=fused_summary, value=fused_summary)
            internal_refined_outputs["text"] = self.norm_t(gated_t + t_refinement)

        if gated_a is not None:
            a_refinement, _ = self.scatter_a_attn(query=gated_a, key=fused_summary, value=fused_summary)
            internal_refined_outputs["audio"] = self.norm_a(gated_a + a_refinement)
            
        gate_info = {"video_gate": w_v, "text_gate": w_t, "audio_gate": w_a}

        # ====================================================================
        # === 步骤 2: 应用可学习的旁路门控，计算最终输出 ==================
        # ====================================================================
        final_outputs = {}

        # 看一下这里往raw还是gated后的偏
        if video_tokens is not None:
            alpha_v = torch.sigmoid(self.bypass_gate_v)
            final_outputs["video"] = (1 - alpha_v) * video_tokens + alpha_v * internal_refined_outputs["video"]
        
        if text_tokens is not None:
            alpha_t = torch.sigmoid(self.bypass_gate_t)
            final_outputs["text"] = (1 - alpha_t) * text_tokens + alpha_t * internal_refined_outputs["text"]

        if audio_tokens is not None:
            alpha_a = torch.sigmoid(self.bypass_gate_a)
            final_outputs["audio"] = (1 - alpha_a) * audio_tokens + alpha_a * internal_refined_outputs["audio"]

        return final_outputs, mu, logvar, gate_info


class SplitAndFuseGatedRouter_wo_BypassGate(nn.Module):
    """
    消融实验 A1: 移除可学习的旁路门控 (Learnable Bypass Gate)。
    
    模块的输出直接是经过内部融合精炼逻辑处理后的结果，
    不再有与原始输入的加权融合。
    """
    def __init__(self, dim: int, num_summary_queries: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # --- 模块定义与最终版本完全相同 ---
        self.dim = dim
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        self.summary_queries = nn.Parameter(torch.randn(num_summary_queries, dim))
        self.gather_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gather_norm1 = nn.LayerNorm(dim)
        self.gather_self_attn_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.to_mu = nn.Linear(dim, dim)
        self.to_logvar = nn.Linear(dim, dim)
        self.scatter_v_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_t_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_a_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 推荐保留这个初始化，让模块内部从一个简单的状态开始学习
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        
        scatter_attns = [
            getattr(m, 'scatter_v_attn', None), getattr(m, 'scatter_t_attn', None), getattr(m, 'scatter_a_attn', None)
        ]
        for attn in scatter_attns:
            if attn is not None and hasattr(attn, 'out_proj'):
                torch.nn.init.zeros_(attn.out_proj.weight)
                if attn.out_proj.bias is not None:
                    torch.nn.init.zeros_(attn.out_proj.bias)

    def _get_global_feature(self, tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tokens is None: return None
        return tokens.mean(dim=1)

    def forward(self, video_tokens=None, text_tokens=None, audio_tokens=None):
        device = next(self.parameters()).device
        batch_size = -1
        for t in [video_tokens, text_tokens, audio_tokens]:
            if t is not None: batch_size = t.shape[0]; break
        if batch_size == -1: raise ValueError("至少需要提供一个模态作为输入。")

        v_global = self._get_global_feature(video_tokens)
        t_global = self._get_global_feature(text_tokens)
        a_global = self._get_global_feature(audio_tokens)

        if v_global is None: v_global = torch.zeros(batch_size, self.dim, device=device)
        if t_global is None: t_global = torch.zeros(batch_size, self.dim, device=device)
        if a_global is None: a_global = torch.zeros(batch_size, self.dim, device=device)

        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1) if video_tokens is not None else None
        gated_t = text_tokens * w_t.unsqueeze(-1) if text_tokens is not None else None
        gated_a = audio_tokens * w_a.unsqueeze(-1) if audio_tokens is not None else None
        
        context_list = [t for t in [gated_v, gated_t, gated_a] if t is not None]
        if not context_list:
            return {k: v for k, v in [("video", video_tokens), ("text", text_tokens), ("audio", audio_tokens)] if v is not None}, None, None, {}
            
        gated_context = torch.cat(context_list, dim=1)

        queries = self.summary_queries.unsqueeze(0).expand(batch_size, -1, -1)
        extracted_info, _ = self.gather_cross_attn(queries, gated_context, gated_context)
        summary = self.gather_norm1(queries + extracted_info)
        fused_summary = self.gather_self_attn_block(summary)

        mu = self.to_mu(fused_summary)
        logvar = self.to_logvar(fused_summary)

        # --- 核心区别: 直接返回内部精炼结果 ---
        refined_outputs = {}
        if gated_v is not None:
            v_refinement, _ = self.scatter_v_attn(query=gated_v, key=fused_summary, value=fused_summary)
            refined_outputs["video"] = self.norm_v(gated_v + v_refinement)
        
        if gated_t is not None:
            t_refinement, _ = self.scatter_t_attn(query=gated_t, key=fused_summary, value=fused_summary)
            refined_outputs["text"] = self.norm_t(gated_t + t_refinement)

        if gated_a is not None:
            a_refinement, _ = self.scatter_a_attn(query=gated_a, key=fused_summary, value=fused_summary)
            refined_outputs["audio"] = self.norm_a(gated_a + a_refinement)
            
        gate_info = {"video_gate": w_v, "text_gate": w_t, "audio_gate": w_a}

        return refined_outputs, mu, logvar, gate_info


class SplitAndFuseGatedRouter_wo_Gate(nn.Module):
    """
    消融实验 A2: 移除模态门控网络 (Gating Network)。
    
    所有模态的输入tokens不经过权重缩放，直接进入后续的融合精炼流程。
    相当于门控权重w_v, w_t, w_a始终为1。
    """
    def __init__(self, dim: int, num_summary_queries: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        # --- 核心区别: 移除了 self.gating_network ---
        self.summary_queries = nn.Parameter(torch.randn(num_summary_queries, dim))
        self.gather_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gather_norm1 = nn.LayerNorm(dim)
        self.gather_self_attn_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.to_mu = nn.Linear(dim, dim)
        self.to_logvar = nn.Linear(dim, dim)
        self.scatter_v_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_t_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_a_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)
    
    # ... _init_weights 和 _get_global_feature 方法保持不变 ...
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        scatter_attns = [
            getattr(m, 'scatter_v_attn', None), getattr(m, 'scatter_t_attn', None), getattr(m, 'scatter_a_attn', None)
        ]
        for attn in scatter_attns:
            if attn is not None and hasattr(attn, 'out_proj'):
                torch.nn.init.zeros_(attn.out_proj.weight)
                if attn.out_proj.bias is not None:
                    torch.nn.init.zeros_(attn.out_proj.bias)
    def _get_global_feature(self, tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tokens is None: return None
        return tokens.mean(dim=1)

    def forward(self, video_tokens=None, text_tokens=None, audio_tokens=None):
        # ... forward 逻辑的前半部分有变化 ...
        device = next(self.parameters()).device
        batch_size = -1
        for t in [video_tokens, text_tokens, audio_tokens]:
            if t is not None: batch_size = t.shape[0]; break
        if batch_size == -1: raise ValueError("至少需要提供一个模态作为输入。")

        # --- 核心区别: 不再计算和应用门控权重 ---
        gated_v = video_tokens
        gated_t = text_tokens
        gated_a = audio_tokens
        
        gate_info = {} # 没有门控信息
        
        # ... 后续逻辑与完整模型一致，除了输入是未加权的 ...
        context_list = [t for t in [gated_v, gated_t, gated_a] if t is not None]
        if not context_list:
            return {k: v for k, v in [("video", video_tokens), ("text", text_tokens), ("audio", audio_tokens)] if v is not None}, None, None, {}
            
        gated_context = torch.cat(context_list, dim=1)

        queries = self.summary_queries.unsqueeze(0).expand(batch_size, -1, -1)
        extracted_info, _ = self.gather_cross_attn(queries, gated_context, gated_context)
        summary = self.gather_norm1(queries + extracted_info)
        fused_summary = self.gather_self_attn_block(summary)

        mu = self.to_mu(fused_summary)
        logvar = self.to_logvar(fused_summary)

        internal_refined_outputs = {}
        if gated_v is not None:
            v_refinement, _ = self.scatter_v_attn(query=gated_v, key=fused_summary, value=fused_summary)
            internal_refined_outputs["video"] = self.norm_v(gated_v + v_refinement)
        
        if gated_t is not None:
            t_refinement, _ = self.scatter_t_attn(query=gated_t, key=fused_summary, value=fused_summary)
            internal_refined_outputs["text"] = self.norm_t(gated_t + t_refinement)

        if gated_a is not None:
            a_refinement, _ = self.scatter_a_attn(query=gated_a, key=fused_summary, value=fused_summary)
            internal_refined_outputs["audio"] = self.norm_a(gated_a + a_refinement)

        final_outputs = {}
        if video_tokens is not None:
            alpha_v = torch.sigmoid(self.bypass_gate_v)
            final_outputs["video"] = (1 - alpha_v) * video_tokens + alpha_v * internal_refined_outputs["video"]
        
        if text_tokens is not None:
            alpha_t = torch.sigmoid(self.bypass_gate_t)
            final_outputs["text"] = (1 - alpha_t) * text_tokens + alpha_t * internal_refined_outputs["text"]

        if audio_tokens is not None:
            alpha_a = torch.sigmoid(self.bypass_gate_a)
            final_outputs["audio"] = (1 - alpha_a) * audio_tokens + alpha_a * internal_refined_outputs["audio"]

        return final_outputs, mu, logvar, gate_info


class SplitAndFuseGatedRouter_wo_Scatter(nn.Module):
    """
    消融实验 A3: 移除分发精炼阶段 (Scatter Phase)。
    
    模型只执行门控和汇聚阶段，得到fused_summary。然后直接返回这个
    fused_summary作为唯一的输出。这本质上是我们最初的“方案二”。
    注意：这将改变模型的输出结构，不再保持原始序列长度。
    """
    def __init__(self, dim: int, num_summary_queries: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        self.summary_queries = nn.Parameter(torch.randn(num_summary_queries, dim))
        self.gather_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gather_norm1 = nn.LayerNorm(dim)
        self.gather_self_attn_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.to_mu = nn.Linear(dim, dim)
        self.to_logvar = nn.Linear(dim, dim)
        
        # --- 核心区别: 移除了所有 scatter 和 bypass 组件 ---

    # ... _get_global_feature 方法保持不变 ...
    def _get_global_feature(self, tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tokens is None: return None
        return tokens.mean(dim=1)

    def forward(self, video_tokens=None, text_tokens=None, audio_tokens=None):
        device = next(self.parameters()).device
        batch_size = -1
        for t in [video_tokens, text_tokens, audio_tokens]:
            if t is not None: batch_size = t.shape[0]; break
        if batch_size == -1: raise ValueError("至少需要提供一个模态作为输入。")
        
        v_global = self._get_global_feature(video_tokens)
        t_global = self._get_global_feature(text_tokens)
        a_global = self._get_global_feature(audio_tokens)

        if v_global is None: v_global = torch.zeros(batch_size, self.dim, device=device)
        if t_global is None: t_global = torch.zeros(batch_size, self.dim, device=device)
        if a_global is None: a_global = torch.zeros(batch_size, self.dim, device=device)

        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1) if video_tokens is not None else None
        gated_t = text_tokens * w_t.unsqueeze(-1) if text_tokens is not None else None
        gated_a = audio_tokens * w_a.unsqueeze(-1) if audio_tokens is not None else None
        
        context_list = [t for t in [gated_v, gated_t, gated_a] if t is not None]
        if not context_list:
            # 如果没有输入，可能需要返回一个特定长度的零向量
            return torch.zeros(batch_size, self.summary_queries.shape[0], self.dim, device=device), None, None, {}

        gated_context = torch.cat(context_list, dim=1)

        queries = self.summary_queries.unsqueeze(0).expand(batch_size, -1, -1)
        extracted_info, _ = self.gather_cross_attn(queries, gated_context, gated_context)
        summary = self.gather_norm1(queries + extracted_info)
        fused_summary = self.gather_self_attn_block(summary)

        mu = self.to_mu(fused_summary)
        logvar = self.to_logvar(fused_summary)
        gate_info = {"video_gate": w_v, "text_gate": w_t, "audio_gate": w_a}
        
        # --- 核心区别: 直接返回融合后的摘要 ---
        # 返回一个张量，而不是字典
        return fused_summary, mu, logvar, gate_info


class SplitAndFuseGatedRouter_wo_Query(nn.Module):
    """
    消融实验 A4: 移除基于查询的汇聚机制 (Query-based Gather)。
    
    使用一个简单的MLP直接从各个模态的全局特征生成fused_summary，
    而不是通过查询向量去主动提取token级细节。
    """
    def __init__(self, dim: int, num_summary_queries: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_summary_queries = num_summary_queries
        
        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        
        # --- 核心区别: 移除 gather 相关的注意力组件 ---
        # 用一个MLP来生成summary
        self.summary_generator = nn.Sequential(
            nn.Linear(dim * 3, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), num_summary_queries * dim) # 输出展平的summary
        )
        
        self.to_mu = nn.Linear(dim, dim)
        self.to_logvar = nn.Linear(dim, dim)
        self.scatter_v_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_t_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.scatter_a_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        
        self.apply(self._init_weights)

    # ... _init_weights 和 _get_global_feature 方法保持不变 ...
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        scatter_attns = [
            getattr(m, 'scatter_v_attn', None), getattr(m, 'scatter_t_attn', None), getattr(m, 'scatter_a_attn', None)
        ]
        for attn in scatter_attns:
            if attn is not None and hasattr(attn, 'out_proj'):
                torch.nn.init.zeros_(attn.out_proj.weight)
                if attn.out_proj.bias is not None:
                    torch.nn.init.zeros_(attn.out_proj.bias)
    def _get_global_feature(self, tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tokens is None: return None
        return tokens.mean(dim=1)

    def forward(self, video_tokens=None, text_tokens=None, audio_tokens=None):
        device = next(self.parameters()).device
        batch_size = -1
        for t in [video_tokens, text_tokens, audio_tokens]:
            if t is not None: batch_size = t.shape[0]; break
        if batch_size == -1: raise ValueError("至少需要提供一个模态作为输入。")

        v_global = self._get_global_feature(video_tokens)
        t_global = self._get_global_feature(text_tokens)
        a_global = self._get_global_feature(audio_tokens)

        if v_global is None: v_global = torch.zeros(batch_size, self.dim, device=device)
        if t_global is None: t_global = torch.zeros(batch_size, self.dim, device=device)
        if a_global is None: a_global = torch.zeros(batch_size, self.dim, device=device)

        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1) if video_tokens is not None else None
        gated_t = text_tokens * w_t.unsqueeze(-1) if text_tokens is not None else None
        gated_a = audio_tokens * w_a.unsqueeze(-1) if audio_tokens is not None else None

        # --- 核心区别: 用MLP生成fused_summary ---
        flat_summary = self.summary_generator(all_global)
        fused_summary = flat_summary.view(batch_size, self.num_summary_queries, self.dim)

        mu = self.to_mu(fused_summary)
        logvar = self.to_logvar(fused_summary)
        gate_info = {"video_gate": w_v, "text_gate": w_t, "audio_gate": w_a}
        
        # ... 后续的分发和旁路逻辑与完整模型一致 ...
        internal_refined_outputs = {}
        if gated_v is not None:
            v_refinement, _ = self.scatter_v_attn(query=gated_v, key=fused_summary, value=fused_summary)
            internal_refined_outputs["video"] = self.norm_v(gated_v + v_refinement)
        
        if gated_t is not None:
            t_refinement, _ = self.scatter_t_attn(query=gated_t, key=fused_summary, value=fused_summary)
            internal_refined_outputs["text"] = self.norm_t(gated_t + t_refinement)

        if gated_a is not None:
            a_refinement, _ = self.scatter_a_attn(query=gated_a, key=fused_summary, value=fused_summary)
            internal_refined_outputs["audio"] = self.norm_a(gated_a + a_refinement)

        final_outputs = {}
        if video_tokens is not None:
            alpha_v = torch.sigmoid(self.bypass_gate_v)
            final_outputs["video"] = (1 - alpha_v) * video_tokens + alpha_v * internal_refined_outputs["video"]
        
        if text_tokens is not None:
            alpha_t = torch.sigmoid(self.bypass_gate_t)
            final_outputs["text"] = (1 - alpha_t) * text_tokens + alpha_t * internal_refined_outputs["text"]

        if audio_tokens is not None:
            alpha_a = torch.sigmoid(self.bypass_gate_a)
            final_outputs["audio"] = (1 - alpha_a) * audio_tokens + alpha_a * internal_refined_outputs["audio"]
            
        return final_outputs, mu, logvar, gate_info

# --- Usage Example ---
if __name__ == '__main__':
    # Config
    B, D, N_QUERIES, N_HEADS = 4, 768, 64, 12
    N_v, N_t, N_a = 256, 77, 100

    # Model
    model = SplitAndFuseGatedRouter(dim=D, num_summary_queries=N_QUERIES, num_heads=N_HEADS)
    print(model)

    # Dummy data
    v_in, t_in, a_in = torch.randn(B, N_v, D), torch.randn(B, N_t, D), torch.randn(B, N_a, D)

    # --- Run forward pass ---
    refined_branches, mu, logvar, gates = model(video_tokens=v_in, text_tokens=t_in, audio_tokens=a_in)

    print("\n--- Output Branch Shapes ---")
    for name, tensor in refined_branches.items():
        print(f"  Refined {name} branch: {tensor.shape}") # Should match input lengths

    # The refined branches can now be concatenated before being fed to the DiT
    final_dit_input = torch.cat(list(refined_branches.values()), dim=1)
    print(f"\nShape of final concatenated input for DiT: {final_dit_input.shape}")

    kl_loss = compute_kl_loss(mu, logvar) # Same KL loss function
    print(f"KL Divergence Loss: {kl_loss.item():.6f}")    