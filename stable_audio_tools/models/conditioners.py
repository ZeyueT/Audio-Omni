#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc
import sys
import importlib.util
from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .utils import copy_state_dict
from .utils import load_ckpt_state_dict

from torch import nn
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import einops
# import h5py
from .SA_transformer_module import SA_Attention, SA_PreNorm, SA_FeedForward
from torchvision import transforms
from omegaconf import OmegaConf
import random
import torch.nn.init as init
from einops import rearrange
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
    
class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)
            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]


import decord
from decord import VideoReader
from decord import cpu
import torch
import math
import einops
import torchvision.transforms as transforms

import os


class SA_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SA_PreNorm(dim, SA_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                SA_PreNorm(dim, SA_FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
       
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, x1, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = x1 // num_heads

        self.query = nn.Linear(x1, x1)
        self.key = nn.Linear(x1, x1)
        self.value = nn.Linear(x1, x1)

        self.final_linear = nn.Linear(x1, x1)

        self.norm1 = nn.LayerNorm(x1)
        self.norm2 = nn.LayerNorm(x1) 
        
        init.constant_(self.final_linear.weight, 0)
        if self.final_linear.bias is not None:
            init.constant_(self.final_linear.bias, 0)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, tensor_A, tensor_B):
        batch_size = tensor_A.size(0)
        
        # tensor_A: [5, 3000, 768]
        # tensor_B: [5, 3200, 768]
        
        Q = self.split_heads(self.query(tensor_A), batch_size) # Q: [5, 3, 3000, 256]
        K = self.split_heads(self.key(tensor_B), batch_size) # K: [5, 3, 3200, 256]
        V = self.split_heads(self.value(tensor_B), batch_size)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5) # attention_scores: [5, 3, 3000, 3200]
        attention_scores = torch.softmax(attention_scores, dim=-1) # attention_scores: [5, 3, 3000, 3200]
        attention_output = torch.matmul(attention_scores, V) # attention_output: [5, 3, 3000, 256]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        output = attention_output.view(batch_size, -1, self.num_heads * self.depth) # [5, 3000, 768]

        output = self.norm1(output + tensor_A)
        output = self.norm2(self.final_linear(output) + output)
        return output


class SA_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SA_PreNorm(dim, SA_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                SA_PreNorm(dim, SA_FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
       
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, x1, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = x1 // num_heads

        self.query = nn.Linear(x1, x1)
        self.key = nn.Linear(x1, x1)
        self.value = nn.Linear(x1, x1)

        self.final_linear = nn.Linear(x1, x1)

        self.norm1 = nn.LayerNorm(x1)
        self.norm2 = nn.LayerNorm(x1) 
        
        init.constant_(self.final_linear.weight, 0)
        if self.final_linear.bias is not None:
            init.constant_(self.final_linear.bias, 0)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, tensor_A, tensor_B):
        batch_size = tensor_A.size(0)
        
        # tensor_A: [5, 3000, 768]
        # tensor_B: [5, 3200, 768]
        
        Q = self.split_heads(self.query(tensor_A), batch_size) # Q: [5, 3, 3000, 256]
        K = self.split_heads(self.key(tensor_B), batch_size) # K: [5, 3, 3200, 256]
        V = self.split_heads(self.value(tensor_B), batch_size)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5) # attention_scores: [5, 3, 3000, 3200]
        attention_scores = torch.softmax(attention_scores, dim=-1) # attention_scores: [5, 3, 3000, 3200]
        attention_output = torch.matmul(attention_scores, V) # attention_output: [5, 3, 3000, 256]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        output = attention_output.view(batch_size, -1, self.num_heads * self.depth) # [5, 3000, 768]

        output = self.norm1(output + tensor_A)
        output = self.norm2(self.final_linear(output) + output)
        return output


class CLIPConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "clip-vit-base-patch32",
            video_fps: int = 5,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000, # 10*10*50 [t, fps, head]
            # in_features: int = 4700, # 47*2*50 [t, fps, head]
            project_out: bool = False,
            mask_ratio: float = 0.0,
            mask_type: str = "input"
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        # sa_depth=8        
        
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        # fps = 20
        fps = 5
        self.clip_model_name='clip-vit-base-patch32'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        
        if self.clip_model_name=='clip-vit-base-patch32':
            if self.mask_type == "input":
                in_features = round(50*(1-self.mask_ratio))*fps*duration
            else:
                in_features = 50*fps*duration
                
            # in_features = int(50*(1-self.mask_ratio))*fps*duration
            out_features = 128
            temporal_dim=768
            model_path = '/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--openai--clip-vit-base-patch32/text_encoder/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)
            # self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
        
            self.in_features = in_features
            self.out_features = out_features

            self.SA_type = 'temporal_SA' # or 'temporal_SA'
            # self.Spa_Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            # self.Spa_Temp_pos_embedding = nn.Parameter(torch.randn(1, 50*10*10, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            if self.SA_type=='temporal_SA':        
                self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration*fps, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            elif self.SA_type=='spatial_SA':
                self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]

            # 获取 CLIP 的标准均值和标准差
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            # 手动在 GPU 上定义预处理管道
            self.preprocess_CLIP = transforms.Compose([
                # transforms.Resize((224, 224)),  # 调整大小
                transforms.Normalize(mean=clip_mean, std=clip_std)  # 归一化
            ])
    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0  # 将像素值缩放到 [0, 1]
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # len_keep = math.ceil(L * (1 - mask_ratio))        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def random_masking_pad_zero(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create a mask with zeros for kept indices and ones for masked indices
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=torch.argsort(ids_shuffle, dim=1))
        
        # Apply mask to the input tensor
        x_masked = x * (1 - mask.unsqueeze(-1))
        
        return x_masked, mask
    def mask_video_tensor(self, Video_tensors, mask_ratio=0.6, patch_size=32):
        """
        将Video_tensors中的每个224x224图像划分为49个32x32的patch，并根据mask_ratio随机选择patch进行mask。
        
        Args:
        - Video_tensors (Tensor): 输入的四维张量，大小为 (A, 3, 224, 224)
        - mask_ratio (float): 需要mask的patch比例，范围在[0,1]之间
        
        Returns:
        - masked_video (Tensor): 掩盖处理后的四维张量
        """
        # 获取输入视频张量的尺寸
        batch_size, channels, height, width = Video_tensors.shape
        
        # patch_size 和 grid_size
        grid_size = height // patch_size
        
        # 生成一个 patch mask
        num_patches = grid_size * grid_size  # 49个patch
        num_patches_to_mask = int(num_patches * mask_ratio)  # 根据mask_ratio计算需要mask的patch个数
        
        # 随机选择需要mask的patch索引
        mask_indices = torch.randperm(num_patches)[:num_patches_to_mask]
        
        # 处理视频的每一帧
        masked_video = Video_tensors.clone()  # 深拷贝原始张量，保留原始数据
        for i in range(batch_size):
            # 对于每一帧图像，进行patch划分和mask
            for idx in mask_indices:
                # 计算该patch的行列位置
                row = idx // grid_size
                col = idx % grid_size
                
                # 获取这个patch的区域范围，填充0
                masked_video[i, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0
                
        return masked_video

        
    def forward(self, Video_tensors: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)
        
        Video_tensors = torch.cat(Video_tensors, dim=0).to(device)
        batch_size, time_length,_,_,_ = Video_tensors.size()

        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')


        if self.mask_type == "input-patch":
            if self.mask_ratio > 0:
                if batch_size==1:
                    self.mask_ratio=0 # infer
                Video_tensors = self.mask_video_tensor(Video_tensors, self.mask_ratio)
        
        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)

        if self.clip_model_name=='clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            if self.mask_ratio > 0:
                class_token_embeddings = video_hidden[:, 0, :]
                # patch_token_embeddings = video_hidden_normal[:, 1:, :]
                if self.mask_type == "input":
                    video_hidden, mask, ids_restore = self.random_masking(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)            
                elif self.mask_type == "input-pad":
                    if batch_size==1: # infer stage
                        self.mask_ratio=0
                    video_hidden, mask = self.random_masking_pad_zero(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)
                    
            if self.SA_type=='temporal_SA':
                # temporal SA
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h',b=batch_size,t=time_length) # [150, 100, 768]  
                video_hidden += self.Temp_pos_embedding
                video_hidden = self.Temp_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h',b=batch_size,t=time_length)
            elif self.SA_type=='spatial_SA':
                # spacial SA
                video_hidden += self.Spatial_pos_embedding
                video_hidden = self.Spatial_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length)

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)
        # zero_matrix = torch.zeros_like(video_hidden)
        # video_hidden = video_hidden * visual_modality_exist + zero_matrix * (1 - visual_modality_exist)
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


class CLIPWithSyncWithEmptyFeatureConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "clip-vit-base-patch32",
            video_fps: int = 5,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000, # 10*10*50 [t, fps, head]
            # in_features: int = 4700, # 47*2*50 [t, fps, head]
            project_out: bool = False,
            mask_ratio: float = 0.0,
            mask_type: str = "input",
            sync_type: str = "add",
            temporal_transformer: bool = True,
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        # sa_depth=8        
        
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        # fps = 20
        print(f"video_fps: {video_fps}")
        fps = video_fps
        self.clip_model_name='clip-vit-base-patch32'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.sync_type = sync_type
        self.temporal_transformer = temporal_transformer
        
        if self.clip_model_name=='clip-vit-base-patch32':
            if self.mask_type == "input":
                in_features = round(50*(1-self.mask_ratio))*fps*duration
            else:
                in_features = 50*fps*duration
                
            # in_features = int(50*(1-self.mask_ratio))*fps*duration
            out_features = 128
            temporal_dim=768
            
            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)
                        
            model_path = '/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--openai--clip-vit-base-patch32/text_encoder/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)
            # self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            
            self.proj_sync = nn.Linear(in_features=240, out_features=out_features)
            if self.sync_type=='add':
                self.sync_weight = nn.Parameter(torch.tensor(0.0))
            elif self.sync_type=="cross-attention":
                cross_attention_num_heads = 3 # MultiHeadCrossAttention
                self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, cross_attention_num_heads)
                
        
            self.in_features = in_features
            self.out_features = out_features

            self.SA_type = 'temporal_SA' # or 'temporal_SA'
            # self.Spa_Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            # self.Spa_Temp_pos_embedding = nn.Parameter(torch.randn(1, 50*10*10, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            if self.temporal_transformer:
                if self.SA_type=='temporal_SA':        
                    self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                    self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration*fps, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
                elif self.SA_type=='spatial_SA':
                    self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                    self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]

            # 获取 CLIP 的标准均值和标准差
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            # 手动在 GPU 上定义预处理管道
            self.preprocess_CLIP = transforms.Compose([
                # transforms.Resize((224, 224)),  # 调整大小
                transforms.Normalize(mean=clip_mean, std=clip_std)  # 归一化
            ])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0  # 将像素值缩放到 [0, 1]
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # len_keep = math.ceil(L * (1 - mask_ratio))        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def random_masking_pad_zero(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create a mask with zeros for kept indices and ones for masked indices
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=torch.argsort(ids_shuffle, dim=1))
        
        # Apply mask to the input tensor
        x_masked = x * (1 - mask.unsqueeze(-1))
        
        return x_masked, mask

                
    def mask_video_tensor(self, Video_tensors, mask_ratio=0.6, patch_size=32):
        """
        将Video_tensors中的每个224x224图像划分为49个32x32的patch，并根据mask_ratio随机选择patch进行mask。
        
        Args:
        - Video_tensors (Tensor): 输入的四维张量，大小为 (A, 3, 224, 224)
        - mask_ratio (float): 需要mask的patch比例，范围在[0,1]之间
        
        Returns:
        - masked_video (Tensor): 掩盖处理后的四维张量
        """
        # 获取输入视频张量的尺寸
        batch_size, channels, height, width = Video_tensors.shape
        
        # patch_size 和 grid_size
        grid_size = height // patch_size
        
        # 生成一个 patch mask
        num_patches = grid_size * grid_size  # 49个patch
        num_patches_to_mask = int(num_patches * mask_ratio)  # 根据mask_ratio计算需要mask的patch个数
        
        # 随机选择需要mask的patch索引
        mask_indices = torch.randperm(num_patches)[:num_patches_to_mask]
        
        # 处理视频的每一帧
        masked_video = Video_tensors.clone()  # 深拷贝原始张量，保留原始数据
        for i in range(batch_size):
            # 对于每一帧图像，进行patch划分和mask
            for idx in mask_indices:
                # 计算该patch的行列位置
                row = idx // grid_size
                col = idx % grid_size
                
                # 获取这个patch的区域范围，填充0
                masked_video[i, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0
                
        return masked_video

    def forward(self, Video_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        Video_tensors = [item["video_tensors"] for item in Video_list]
        video_sync_frames = [item["video_sync_frames"] for item in Video_list]
        video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)
        
        original_videos = torch.cat(Video_tensors, dim=0)
        batch_size, time_length, _, _, _ = original_videos.size()
        # 记录哪些样本全为0，得到一个布尔张量 shape: (B,)
        is_zero = torch.all(original_videos == 0, dim=(1,2,3,4))
        # 后续继续使用 original_videos 作为视频数据
        Video_tensors = original_videos
        
        # 创建一个全零矩阵，形状与 video_hidden 相同
        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')


        if self.mask_type == "input-patch":
            if self.mask_ratio > 0:
                if batch_size==1:
                    self.mask_ratio=0 # infer
                Video_tensors = self.mask_video_tensor(Video_tensors, self.mask_ratio)
        
        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)

        if self.clip_model_name=='clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            if self.mask_ratio > 0:
                class_token_embeddings = video_hidden[:, 0, :]
                # patch_token_embeddings = video_hidden_normal[:, 1:, :]
                if self.mask_type == "input":
                    video_hidden, mask, ids_restore = self.random_masking(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)            
                elif self.mask_type == "input-pad":
                    if batch_size==1: # infer stage
                        self.mask_ratio=0
                    video_hidden, mask = self.random_masking_pad_zero(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)
            
            if self.temporal_transformer:
                if self.SA_type=='temporal_SA':
                    # temporal SA
                    video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h',b=batch_size,t=time_length) # [150, 100, 768]  
                    video_hidden += self.Temp_pos_embedding
                    video_hidden = self.Temp_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                    video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h',b=batch_size,t=time_length)
                elif self.SA_type=='spatial_SA':
                    # spacial SA
                    video_hidden += self.Spatial_pos_embedding
                    video_hidden = self.Spatial_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                    video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length)
            else:
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length) # [150, 100, 768]  

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        video_sync_frames = self.proj_sync(video_sync_frames.view(-1, 240))
        video_sync_frames = video_sync_frames.view(batch_size, self.out_features, -1)
        
        if self.sync_type=='add':
            video_hidden = video_hidden + self.sync_weight * video_sync_frames
        elif self.sync_type=='cross-attention':
            video_hidden = self.multi_head_cross_attention(video_hidden, video_sync_frames)
        
        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden) 
        # print('with sync: empty_visual_feat.shape: ', empty_visual_feat.shape)               
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class GatedResidualQueryConnector(nn.Module):
    """
    一个受MAF_Block启发的、用于精炼omni_features的Connector。

    工作流程:
    1. 宏观门控: 对输入的omni_features进行逐个token的加权筛选。
    2. 查询与更新: 用一组可学习的query，通过Cross-Attention“总结”门控后的特征。
    3. 内部融合: 更新后的query在一个共享的Transformer中进行深度融合（自注意力）。
    4. 广播与残差: 将融合后的信息（取平均）广播，并通过一个可学习的门控，
       以残差形式添加到原始的、经过归一化的omni_features上。
    """
    def __init__(self, dim: int, query_len: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.query_len = query_len
        
        # 1. 宏观门控网络 (作用于每个token)
        self.gating_network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # 2. 可学习的Query (统一专家团队)
        self.query_tokens = nn.Parameter(torch.randn(1, query_len, dim))

        # 3. 查询、更新与融合组件
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # 用于内部融合的Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # 4. 广播与残差组件
        self.norm2 = nn.LayerNorm(dim)
        # 可学习的旁路门控参数，初始化为一个很小的值，以实现“平滑过渡”
        self.bypass_gate = nn.Parameter(torch.tensor(-10.0)) 
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 零初始化交叉注意力的输出投影，有助于稳定训练初期
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, 'out_proj'):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
            self,
            omni_features: torch.Tensor,
            omni_mask: torch.Tensor, # True for padding
        ) -> torch.Tensor:
        
        batch_size, long_len, dim = omni_features.shape

        # --- (宏观门控，查询与更新，内部融合 的部分保持不变) ---
        
        # 1. 门控
        gate_values = self.gating_network(omni_features)
        gated_omni = omni_features * gate_values

        # 2. 查询与更新
        queries = self.query_tokens.expand(batch_size, -1, -1)
        summary, _ = self.cross_attn(queries, gated_omni, gated_omni, key_padding_mask=omni_mask)
        updated_queries = self.norm1(queries + summary)
        
        # 3. 内部融合
        fused_queries = self.fusion_transformer(updated_queries)
        
        # --- 4. ** 修正后的、安全的广播与残差 ** ---
        
        # a. 计算全局精炼信号
        refinement_signal = fused_queries.mean(dim=1) # -> [B, D]

        # b. 获取门控权重
        alpha = torch.sigmoid(self.bypass_gate)

        # c. 将精炼信号广播并归一化
        #    unsqueeze(1) -> [B, 1, D]
        broadcasted_refinement = self.norm2(refinement_signal).unsqueeze(1)
        
        # d. ** 关键的“安全”添加步骤 **
        #    创建一个与 omni_features 形状相同的、包含广播信号的Tensor
        refinement_map = broadcasted_refinement.expand(-1, long_len, -1) # -> [B, long_len, D]
        
        safe_refinement_map = refinement_map.masked_fill(omni_mask.unsqueeze(-1).logical_not(), 0.0)

        # e. 执行残差连接
        final_features = omni_features + alpha * safe_refinement_map
        
        
        return final_features


# 消融. pad feature到2056，加上mask，传递到dit
class QwenOmniWithSyncWithEmptyFeatureConditioner(Conditioner):
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
        # from qwen_omni_utils import process_mm_info
        self.pad_length = pad_length


        if zip_features:
            self.zip_features = True
            self.proj_omni_seq_len = nn.Linear(in_features=pad_length, out_features=zip_length)

        _qwen_model_path = os.environ.get(
            "QWEN_OMNI_MODEL_PATH",
            "/mnt/shangcephfs/mm-base-vision-ascend/zeyuetian/AudioX-private/model/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e",
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


        self.connector = connector
        if connector:
            print("INFO: Initializing GatedResidualQueryConnector.")
            # 从config中获取参数，或使用默认值
            connector_config = connector_config or {}
            self.connector = GatedResidualQueryConnector(
                dim=qwen_feature_dim, # Connector的输入是原始的omni特征
                query_len=connector_config.get("query_len", 128),           # Connector的输出维度需要匹配DiT的期望
                num_heads=connector_config.get("num_heads", 32), # e.g., 2048/64=32
                num_fusion_layers=connector_config.get("num_fusion_layers", 4),
            )
            # proj_features不再需要，因为Connector接管了投影功能

            # 打印参数量
            total_params = sum(p.numel() for p in self.connector.parameters())
            trainable_params = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
            print(f"GatedResidualQueryConnector parameters:")
            print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
            print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
            print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
            
            # 打印各模块的参数量
            print(f"  Breakdown by module:")
            for name, module in self.connector.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                print(f"    {name}: {module_params:,} ({module_params / 1e6:.2f}M)")


        self.norm = RMSNorm(qwen_feature_dim)

    def forward(self, prompt_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        pad_length = self.pad_length
        batch_size = len(prompt_list)
        if isinstance(prompt_list[0], dict):
            # qwen_omni_input_list = [item["qwen_omni_input"] for item in prompt_list]
            # sync_feature_list = [item["sync_feature"] for item in prompt_list]        

            text_prompts =  [item.get("text_prompt") for item in prompt_list]
            video_prompts = [item.get("video_prompt") for item in prompt_list]
            audio_prompts = [item.get("audio_prompt") for item in prompt_list]
            
            # 移除列表中的 None，如果列表全为 None 或为空则传入 None 给 processor
            text_prompts = [p for p in text_prompts if p is not None] or None
            # 对于 video_prompts，需要额外检查：过滤掉 None 和空列表/空字符串等无效值
            video_prompts_filtered = []
            for p in video_prompts:
                if p is not None and p.sum() != 0:
                    p = p.to('cpu')
                    # 检查是否是有效的视频数据（不是空列表、空字符串等）
                    if isinstance(p, (list, tuple)) and len(p) > 0:
                        video_prompts_filtered.append(p)
                    elif isinstance(p, torch.Tensor) and p.numel() > 0:
                        video_prompts_filtered.append(p)
                    elif isinstance(p, str) and p.strip():
                        video_prompts_filtered.append(p)
                    elif not isinstance(p, (list, tuple, str, torch.Tensor)):
                        # 其他类型（如 numpy array 等）也保留
                        video_prompts_filtered.append(p)
            video_prompts = video_prompts_filtered if video_prompts_filtered else None
            audio_prompts = [p for p in audio_prompts if p is not None and p.sum() > 0] or None
            
            # inputs = self.processor(text=text_prompts, audio=audio_prompts, images=None, videos=video_prompts, return_tensors="pt", padding=True, videos_kwargs=self.videos_kwargs)
            inputs = self.processor(text=text_prompts, audio=audio_prompts, images=None, videos=video_prompts,  return_tensors="pt", padding=True, use_audio_in_video=False)
            qwen_mask = inputs.get("attention_mask", None)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            # text_prompt = self.processor.batch_decode(inputs['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # assert batch_size == 1, "batch_size must be 1"
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度，使用 float32
                    qwen_omni_output = self.model.thinker(**inputs,output_hidden_states=True,return_dict=True)
                    omni_features = qwen_omni_output.hidden_states[self.layer_idx].bfloat16() # omni_features: [B, seq_len, hidden_dim]
            # 使用 qwen_mask 提取真实的 feature 部分（去掉 pad 位置）
            if qwen_mask is not None:
                qwen_mask = qwen_mask.to(device)
                # 确保序列长度匹配
                seq_len = omni_features.shape[1]
                mask_seq_len = qwen_mask.shape[1]
                if mask_seq_len != seq_len:
                    if mask_seq_len > seq_len:
                        qwen_mask = qwen_mask[:, :seq_len]
                    else:
                        # 如果 mask 更短，用 0 填充（表示 pad 部分）
                        pad_length_mask = seq_len - mask_seq_len
                        qwen_mask = F.pad(qwen_mask, (0, pad_length_mask), value=0)
                
                # 提取真实的 feature：只保留 mask=1 的位置
                batch_size = omni_features.shape[0]
                real_features_list = []
                real_lengths = []
                
                for b in range(batch_size):
                    batch_mask = qwen_mask[b].bool()  # [seq_len]
                    batch_features = omni_features[b]  # [seq_len, hidden_dim]
                    # 只保留真实位置的特征（去掉 pad 部分）
                    real_features = batch_features[batch_mask]  # [real_len, hidden_dim]
                    real_features_list.append(real_features)
                    real_lengths.append(real_features.shape[0])
                # 直接对每个 real_features 截断 / pad 到 pad_length
                padded_features_list = []
                for real_features in real_features_list:
                    real_len = real_features.shape[0]

                    # 如果真实长度超过 pad_length，使用插值压缩（而不是直接截断）
                    if real_len > pad_length:
                        print(f"[QwenOmniWithSyncWithEmptyFeatureConditioner] Interpolating real_len {real_len} to pad_length {pad_length}")
                        # 使用插值平滑压缩：将 [real_len, hidden_dim] 压缩到 [pad_length, hidden_dim]
                        # 转置为 [hidden_dim, real_len]，然后插值到 [hidden_dim, pad_length]
                        real_features = real_features.transpose(0, 1)  # [hidden_dim, real_len]
                        real_features = F.interpolate(
                            real_features.unsqueeze(0),  # [1, hidden_dim, real_len]
                            size=pad_length,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0)  # [hidden_dim, pad_length]
                        real_features = real_features.transpose(0, 1)  # [pad_length, hidden_dim]
                        real_len = pad_length

                    # 右 padding 到 pad_length（如果不足）
                    if real_len < pad_length:
                        padded_features = F.pad(real_features, (0, 0, 0, pad_length - real_len), value=0)
                    else:
                        padded_features = real_features

                    padded_features_list.append(padded_features)
                # 重新堆叠成 tensor
                omni_features = torch.stack(padded_features_list, dim=0)  # [B, pad_length, hidden_dim]
            else:
                # 如果没有 qwen_mask，使用全部 features
                orig_len = omni_features.shape[1]
                real_lengths = [orig_len] * omni_features.shape[0]
                try:
                    assert pad_length >= orig_len, f"pad_length ({pad_length}) must be >= omni_features length ({orig_len})"
                except:
                    print(f"pad_length ({pad_length}) must be >= omni_features length ({orig_len})")
                # 对 omni_features 在时间维度 pad 到 pad_length（右 padding）
                omni_features = F.pad(omni_features, (0, 0, 0, pad_length - orig_len), value=0)
            # 构造与 omni_features 对齐的 attention_mask：
            # 前 orig_len 为有效 token（1 / True），后面的 pad 部分为 0 / False
            if qwen_mask is not None:
                # 基于真实长度构造 mask
                attention_mask = torch.zeros(omni_features.shape[0], pad_length, dtype=torch.bool, device=omni_features.device)
                for b, real_len in enumerate(real_lengths):
                    attention_mask[b, :real_len] = True
            else:
                # 如果没有 qwen_mask，创建全为 True 的 mask
                attention_mask = torch.ones(omni_features.shape[0], orig_len, dtype=torch.bool, device=omni_features.device)
                attention_mask = F.pad(attention_mask, (0, pad_length - orig_len), value=False)


        elif isinstance(prompt_list[0], torch.Tensor):
            # 优化版本：减少不必要的操作，使用向量化创建 mask
            padded_features_list = []
            real_lengths = []

            for feature_tensor in prompt_list:
                feature_tensor = feature_tensor.to(device)
                # 统一为 [seq_len, hidden_dim] 格式（减少维度转换）
                if feature_tensor.dim() == 3:
                    feature_tensor = feature_tensor[0]  # 直接索引，比 squeeze 更快

                orig_len = feature_tensor.shape[0]
                real_len = min(orig_len, pad_length)
                real_lengths.append(real_len)
                
                # 直接 pad/truncate 到 pad_length（一步到位）
                if orig_len < pad_length:
                    padded_feature = F.pad(feature_tensor, (0, 0, 0, pad_length - orig_len), value=0)
                elif orig_len > pad_length:
                    padded_feature = feature_tensor[:pad_length]
                else:
                    padded_feature = feature_tensor
                
                padded_features_list.append(padded_feature)

            # 批量 stack: [B, pad_length, hidden_dim]
            omni_features = torch.stack(padded_features_list, dim=0)

            # 向量化创建 attention_mask（避免 Python 循环，使用 GPU 加速）
            seq_indices = torch.arange(pad_length, device=omni_features.device, dtype=torch.long).unsqueeze(0)  # [1, pad_length]
            real_lengths_tensor = torch.tensor(real_lengths, device=omni_features.device, dtype=torch.long).unsqueeze(1)  # [B, 1]
            attention_mask = seq_indices < real_lengths_tensor  # [B, pad_length] - 向量化操作
        else:
            raise ValueError(f"Unknown prompt type: {type(prompt_list[0])}")


        omni_features = self.norm(omni_features)
        if not self.connector:
            omni_features = self.proj_features(omni_features.to(self.proj_features.weight.dtype))
        else:
            omni_features = self.connector(omni_features, attention_mask)
            omni_features = self.proj_features(omni_features.to(self.proj_features.weight.dtype))

        if hasattr(self, "zip_features") and self.zip_features:
            omni_features = self.proj_omni_seq_len(omni_features.transpose(1, 2)).transpose(1, 2)

        return omni_features, attention_mask


# MetaQuery with qwen-omni (Pure MetaQuery Style - inject tokens in tokenizer)
class MetaQueryWithQwenOmniConditioner(Conditioner):
    """
    Pure MetaQuery-style conditioner for Qwen-Omni:
    - Extend tokenizer with <begin_of_img>, <end_of_img>, <img0>...<imgN>
    - Inject these tokens into input_ids during tokenization
    - Extract their hidden states from Qwen-Omni output
    - Pass through a connector (transformer encoder + MLP) to get final condition
    """

    QWEN_VL_MODELS = ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-3B"]

    def __init__(
        self,
        output_dim: int,
        qwen_omni_model_name: str = "Qwen/Qwen2.5-Omni-3B",
        add_sync_feature: bool = False,
        audio_encoder_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
        video_fps: int = 5,
        project_out: bool = False,
        in_features: int = 1619,
        out_features: int = 128,
        num_metaqueries: int = 64,
        # connector_num_hidden_layers: int = 4,
        connector_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        assert (
            qwen_omni_model_name in self.QWEN_VL_MODELS
        ), f"Unknown QwenVL model name: {qwen_omni_model_name}"
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        self.qwen_omni_model_name = qwen_omni_model_name
        self.video_fps = video_fps
        self.num_metaqueries = num_metaqueries
        # self.connector_num_hidden_layers = connector_num_hidden_layers
        
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        model_path = "/mnt/shangcephfs/mm-base-vision-ascend/zeyuetian/AudioX-private/model/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"
        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        # Get original vocab size before extension
        self.num_embeddings = self.model.thinker.get_input_embeddings().num_embeddings

        # Extend embeddings and tokenizer with MetaQuery tokens
        if self.num_metaqueries > 0:

            
            # Resize embeddings: +2 for <begin_of_img> and <end_of_img>
            self.model.thinker.resize_token_embeddings(
                self.num_embeddings + self.num_metaqueries + 2
            )

            # Add special tokens to tokenizer
            tokenizer = self.processor.tokenizer
            new_tokens = ["<begin_of_img>", "<end_of_img>"] + [
                f"<img{i}>" for i in range(self.num_metaqueries)
            ]
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

            # Store token IDs for extraction
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")
            # Register hook: freeze original embeddings, only train new MetaQuery embeddings
            def freeze_hook(grad):
                grad[:self.num_embeddings].zero_()
                return grad
            
            self.model.thinker.get_input_embeddings().weight.register_hook(freeze_hook)
            
            self.suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(self.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )

            print(f"MetaQuery: Extended tokenizer with {self.num_metaqueries} tokens")
            print(f"  boi_token_id={self.boi_token_id}, eoi_token_id={self.eoi_token_id}")

        # ---- Precision trick: keep Qwen-Omni model in bf16, but train embeddings in fp32 ----
        import os as _os
        if _os.environ.get("QWEN_EMBED_FP32", "1").strip().lower() in ("1", "true", "yes", "y", "on"):
            try:
                _emb = self.model.thinker.get_input_embeddings()
                # Make embedding weights fp32 (trainable)
                _emb.to(dtype=torch.float32)
                # Ensure downstream stays bf16: cast embedding activations back to bf16
                _orig_forward = _emb.forward

                def _forward_fp32_weight_bf16_out(input_ids):
                    out = _orig_forward(input_ids)
                    return out.to(dtype=torch.bfloat16)

                _emb.forward = _forward_fp32_weight_bf16_out
                print("[precision] Qwen thinker embeddings set to fp32 (trainable); embedding outputs cast to bf16.")
            except Exception as _e:
                print(f"[precision] Failed to set Qwen embeddings fp32: {_e}")
        
        # Freeze Qwen-Omni except the extended embeddings
        for name, param in self.model.named_parameters():
            if 'embed_tokens' in name:
                param.requires_grad = True  # Only train new embeddings
            else:
                param.requires_grad = False
        
        # Move components except thinker to CPU
        for name, module in self.model.named_children():
            if name != 'thinker':
                module.cpu()
        print('Qwen-Omni loaded. Only thinker kept on GPU.')

        # Determine Qwen hidden size
        if qwen_omni_model_name == "Qwen/Qwen2.5-Omni-7B":
            qwen_hidden_size = 3584
        elif qwen_omni_model_name == "Qwen/Qwen2.5-Omni-3B":
            qwen_hidden_size = 2048
        
        # Connector: Use Qwen2.5-Omni decoder layers (bidirectional encoder style)
        # from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniDecoderLayer
        from transformers import (
                LlavaOnevisionForConditionalGeneration,
                Qwen2_5_VLForConditionalGeneration,
                Qwen2Config,
            )
        from .transformer_encoder import Qwen2Encoder
        self.connector_in_dim = connector_config["connector_in_dim"]
        self.connector_out_dim = connector_config["connector_out_dim"]
        self.connector_num_hidden_layers = connector_config["connector_num_hidden_layers"]
        
        connector_norm = RMSNorm(self.connector_out_dim, eps=1e-5)
        # with torch.no_grad():
        #     connector_norm.weight.fill_(input_scale)            
        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=self.connector_num_hidden_layers,
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            connector_norm,
        )
        if not isinstance(self.connector, nn.Identity):
            for module in self.connector:
                if isinstance(module, Qwen2Encoder):
                    module.gradient_checkpointing_enable({"use_reentrant": False})

        self.videos_kwargs = {
            "do_resize": False,
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
        }          
        print(f"MetaQuery Connector: {self.connector_num_hidden_layers} Qwen2.5-Omni layers, {self.connector_in_dim} -> {self.connector_out_dim}")

    # for speech
    def forward(
        self,
        prompt_list: tp.List[torch.Tensor],
        device: tp.Union[torch.device, str],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (Pure MetaQuery style):
        1. Inject MetaQuery tokens into input_ids
        2. Run through Qwen-Omni thinker to get hidden states
        3. Extract the segment between <begin_of_img> and <end_of_img>
        4. Pass through connector (Qwen2.5-Omni layers + MLP + norm)
        5. Return as condition for DiT
        """
        batch_size = len(prompt_list)
        # assert batch_size == 1, "Currently only supports batch_size=1"

        text_prompts =  [item.get("text_prompt") for item in prompt_list]
        video_prompts = [item.get("video_prompt") for item in prompt_list]
        audio_prompts = [item.get("audio_prompt") for item in prompt_list]

        text_prompts = [p + self.suffix for p in text_prompts]

        # 移除列表中的 None，如果列表全为 None 或为空则传入 None 给 processor
        text_prompts = [p for p in text_prompts if p is not None] or None
        # 对于 video_prompts，需要额外检查：过滤掉 None 和空列表/空字符串等无效值
        video_prompts_filtered = []
        for p in video_prompts:
            if p is not None:
                p = p.to('cpu')
                # 检查是否是有效的视频数据（不是空列表、空字符串等）
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    video_prompts_filtered.append(p)
                elif isinstance(p, torch.Tensor) and p.numel() > 0:
                    video_prompts_filtered.append(p)
                elif isinstance(p, str) and p.strip():
                    video_prompts_filtered.append(p)
                elif not isinstance(p, (list, tuple, str, torch.Tensor)):
                    # 其他类型（如 numpy array 等）也保留
                    video_prompts_filtered.append(p)
        video_prompts = video_prompts_filtered if video_prompts_filtered else None
        audio_prompts = [p for p in audio_prompts if p is not None] or None
        inputs = self.processor(text=text_prompts, audio=audio_prompts, images=None, videos=video_prompts, return_tensors="pt", padding=True, videos_kwargs=self.videos_kwargs)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        with torch.cuda.amp.autocast(enabled=False):
            qwen_omni_output = self.model.thinker(
                **inputs, output_hidden_states=True, return_dict=True
            )
            prompt_embeds = qwen_omni_output.hidden_states[-2]  # [B, L, hidden_size]
        # === Step 3: Extract MetaQuery segment ===
        if self.num_metaqueries > 0:
            boi_pos = torch.where(input_ids == self.boi_token_id)[1]
            eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]
            # Create mask for tokens between BOI and EOI (exclusive)
            B, L = input_ids.shape
            indices = torch.arange(L, device=input_ids.device)[None, :].expand(B, -1)
            mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
            # Extract MetaQuery hidden states
            metaquery_embeds = prompt_embeds[mask].view(B, -1, prompt_embeds.size(-1))
            if attention_mask is not None:
                metaquery_attention_mask = attention_mask[mask].view(B, -1)
            else:
                metaquery_attention_mask = None
        else:
            # Fallback: use all hidden states
            metaquery_embeds = prompt_embeds
            metaquery_attention_mask = attention_mask

        # === Step 4: Pass through connector (Qwen2.5-Omni decoder layers) ===
        metaquery_features = metaquery_embeds

        metaquery_features = self.connector(metaquery_features)
        # metaquery_features = self.connector_norm(metaquery_features)

        # Return: [B, num_metaqueries, output_dim] + mask
        return metaquery_features, torch.ones(metaquery_features.shape[0], metaquery_features.shape[1]).to(device)


class CLIPWithSyncConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "clip-vit-base-patch32",
            video_fps: int = 15,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000, # 10*10*50 [t, fps, head]
            # in_features: int = 4700, # 47*2*50 [t, fps, head]
            project_out: bool = False,
            mask_ratio: float = 0.0,
            mask_type: str = "input",
            sync_type: str = "add"
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        # sa_depth=8        
        
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        # fps = 20
        fps = 15
        # fps = video_fps
        self.clip_model_name='clip-vit-base-patch32'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.sync_type = sync_type
        
        if self.clip_model_name=='clip-vit-base-patch32':
            if self.mask_type == "input":
                in_features = round(50*(1-self.mask_ratio))*fps*duration
            else:
                in_features = 50*fps*duration
                
            # in_features = int(50*(1-self.mask_ratio))*fps*duration
            out_features = 128
            temporal_dim=768
            model_path = '/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--openai--clip-vit-base-patch32/text_encoder/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)
            # self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            
            self.proj_sync = nn.Linear(in_features=240, out_features=out_features)
            if self.sync_type=='add':
                self.sync_weight = nn.Parameter(torch.tensor(0.0))
            elif self.sync_type=="cross-attention":
                cross_attention_num_heads = 3 # MultiHeadCrossAttention
                self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, cross_attention_num_heads)
                
        
            self.in_features = in_features
            self.out_features = out_features

            self.SA_type = 'temporal_SA' # or 'temporal_SA'
            # self.Spa_Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            # self.Spa_Temp_pos_embedding = nn.Parameter(torch.randn(1, 50*10*10, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            if self.SA_type=='temporal_SA':        
                self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration*fps, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            elif self.SA_type=='spatial_SA':
                self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]

            # 获取 CLIP 的标准均值和标准差
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            # 手动在 GPU 上定义预处理管道
            self.preprocess_CLIP = transforms.Compose([
                # transforms.Resize((224, 224)),  # 调整大小
                transforms.Normalize(mean=clip_mean, std=clip_std)  # 归一化
            ])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0  # 将像素值缩放到 [0, 1]
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # len_keep = math.ceil(L * (1 - mask_ratio))        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def random_masking_pad_zero(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create a mask with zeros for kept indices and ones for masked indices
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=torch.argsort(ids_shuffle, dim=1))
        
        # Apply mask to the input tensor
        x_masked = x * (1 - mask.unsqueeze(-1))
        
        return x_masked, mask

                
    def mask_video_tensor(self, Video_tensors, mask_ratio=0.6, patch_size=32):
        """
        将Video_tensors中的每个224x224图像划分为49个32x32的patch，并根据mask_ratio随机选择patch进行mask。
        
        Args:
        - Video_tensors (Tensor): 输入的四维张量，大小为 (A, 3, 224, 224)
        - mask_ratio (float): 需要mask的patch比例，范围在[0,1]之间
        
        Returns:
        - masked_video (Tensor): 掩盖处理后的四维张量
        """
        # 获取输入视频张量的尺寸
        batch_size, channels, height, width = Video_tensors.shape
        
        # patch_size 和 grid_size
        grid_size = height // patch_size
        
        # 生成一个 patch mask
        num_patches = grid_size * grid_size  # 49个patch
        num_patches_to_mask = int(num_patches * mask_ratio)  # 根据mask_ratio计算需要mask的patch个数
        
        # 随机选择需要mask的patch索引
        mask_indices = torch.randperm(num_patches)[:num_patches_to_mask]
        
        # 处理视频的每一帧
        masked_video = Video_tensors.clone()  # 深拷贝原始张量，保留原始数据
        for i in range(batch_size):
            # 对于每一帧图像，进行patch划分和mask
            for idx in mask_indices:
                # 计算该patch的行列位置
                row = idx // grid_size
                col = idx % grid_size
                
                # 获取这个patch的区域范围，填充0
                masked_video[i, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0
                
        return masked_video

    def forward(self, Video_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        Video_tensors = [item["video_tensors"] for item in Video_list]
        video_sync_frames = [item["video_sync_frames"] for item in Video_list]

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)
        
        Video_tensors = torch.cat(Video_tensors, dim=0).to(device)
        video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)
        
        batch_size, time_length,_,_,_ = Video_tensors.size()

        # 创建一个全零矩阵，形状与 video_hidden 相同
        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')


        if self.mask_type == "input-patch":
            if self.mask_ratio > 0:
                if batch_size==1:
                    self.mask_ratio=0 # infer
                Video_tensors = self.mask_video_tensor(Video_tensors, self.mask_ratio)
        
        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)

        if self.clip_model_name=='clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            if self.mask_ratio > 0:
                class_token_embeddings = video_hidden[:, 0, :]
                # patch_token_embeddings = video_hidden_normal[:, 1:, :]
                if self.mask_type == "input":
                    video_hidden, mask, ids_restore = self.random_masking(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)            
                elif self.mask_type == "input-pad":
                    if batch_size==1: # infer stage
                        self.mask_ratio=0
                    video_hidden, mask = self.random_masking_pad_zero(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)
                    
            if self.SA_type=='temporal_SA':
                # temporal SA
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h',b=batch_size,t=time_length) # [150, 100, 768]  
                video_hidden += self.Temp_pos_embedding
                video_hidden = self.Temp_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h',b=batch_size,t=time_length)
            elif self.SA_type=='spatial_SA':
                # spacial SA
                video_hidden += self.Spatial_pos_embedding
                video_hidden = self.Spatial_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length)

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        video_sync_frames = self.proj_sync(video_sync_frames.view(-1, 240))
        video_sync_frames = video_sync_frames.view(batch_size, self.out_features, -1)
        
        if self.sync_type=='add':
            video_hidden = video_hidden + self.sync_weight * video_sync_frames
        elif self.sync_type=='cross-attention':
            video_hidden = self.multi_head_cross_attention(video_hidden, video_sync_frames)
        
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


from .synchformer.motionformer import MotionFormer
from typing import Any, Mapping

class SynchformerConditioner(Conditioner):
    def __init__(self, sync_fps: int = 32, sync_seq_dim: int = 240, sync_output_dim: int = 128, input_dim: int=768, output_dim: int = 1536, project_out: bool = True):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        self.dim_project = nn.Linear(sync_seq_dim, sync_output_dim)
        # self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.empty_sync_feature = torch.zeros(1, 240, 768)

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)
        torch.nn.init.constant_(self.proj.weight, 0.)
        torch.nn.init.constant_(self.proj.bias, 0.)

    def forward(self, Video_sync_features: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.empty_sync_feature = self.empty_sync_feature.to(device=device, dtype=Video_sync_features[0].dtype)

        Video_sync_features = torch.cat(Video_sync_features, dim=0).to(device)
        # 检测哪些 batch 的值全为 0
        is_zero_batch = (Video_sync_features.abs().sum(dim=(1, 2)) == 0)  # shape: (b,)
        
        # 对于全零的 batch，用 empty_sync_feature 替换
        Video_sync_features[is_zero_batch] = self.empty_sync_feature

        Video_sync_features = self.norm(Video_sync_features)
        Video_sync_features = self.proj(Video_sync_features)


        Video_sync_features = einops.rearrange(Video_sync_features, 'b t c -> b c t')
        Video_sync_features = self.dim_project(Video_sync_features)
        Video_sync_features = einops.rearrange(Video_sync_features, 'b c t -> b t c')
        return Video_sync_features, torch.ones(Video_sync_features.shape[0], Video_sync_features.shape[1]).to(device)

                   
class DINOv2EncoderConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32", "dinov2-base"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "dinov2-base",
            video_fps: int = 5,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000, # 10*10*50 [t, fps, head]
            # in_features: int = 4700, # 47*2*50 [t, fps, head]
            project_out: bool = False,
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        # sa_depth=8        
        
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        # fps = 20
        print(f"video_fps: {video_fps}")
        fps = video_fps
        self.clip_model_name='dinov2-base'
            
        if self.clip_model_name=='dinov2-base':
            out_features = 128
            temporal_dim=768
            
            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)

            from transformers import AutoImageProcessor, AutoModel   
            model_path = '/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415'
            self.visual_encoder_model = AutoModel.from_pretrained(model_path)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            # self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            
        
            self.in_features = in_features
            self.out_features = out_features

            self.SA_type = 'temporal_SA' # or 'temporal_SA'
            # self.Spa_Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            # self.Spa_Temp_pos_embedding = nn.Parameter(torch.randn(1, 50*10*10, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            if self.SA_type=='temporal_SA':        
                self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration*fps, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            elif self.SA_type=='spatial_SA':
                self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]


    def forward(self, Video_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        Video_tensors = Video_list

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)
        
        # Video_tensors = torch.cat(Video_tensors, dim=0).to(device)
        # batch_size, time_length,_,_,_ = Video_tensors.size()
        
        
        original_videos = torch.cat(Video_tensors, dim=0)
        batch_size, time_length, _, _, _ = original_videos.size()
        # 记录哪些样本全为0，得到一个布尔张量 shape: (B,)
        is_zero = torch.all(original_videos == 0, dim=(1,2,3,4))
        # 后续继续使用 original_videos 作为视频数据
        Video_tensors = original_videos
        
        # 创建一个全零矩阵，形状与 video_hidden 相同
        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')


        # video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)
        video_cond_pixel_values = self.processor(images=Video_tensors.to(device), return_tensors="pt").to(device)


        if self.clip_model_name=='dinov2-base':
            with torch.no_grad():
                outputs = visual_encoder_model(**video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state


            if self.SA_type=='temporal_SA':
                # temporal SA
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h',b=batch_size,t=time_length) # [150, 100, 768]  
                video_hidden += self.Temp_pos_embedding
                video_hidden = self.Temp_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h',b=batch_size,t=time_length)
            elif self.SA_type=='spatial_SA':
                # spacial SA
                video_hidden += self.Spatial_pos_embedding
                video_hidden = self.Spatial_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length)

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden) 
        # print('with sync: empty_visual_feat.shape: ', empty_visual_feat.shape)               
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


class CLIPWithSyncWithoutEmptyFeatureConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "clip-vit-base-patch32",
            video_fps: int = 5,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000, # 10*10*50 [t, fps, head]
            # in_features: int = 4700, # 47*2*50 [t, fps, head]
            project_out: bool = False,
            mask_ratio: float = 0.0,
            mask_type: str = "input",
            sync_type: str = "add"
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        # sa_depth=8        
        
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        # fps = 20
        print(f"video_fps: {video_fps}")
        fps = video_fps
        self.clip_model_name='clip-vit-base-patch32'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.sync_type = sync_type
        
        if self.clip_model_name=='clip-vit-base-patch32':
            if self.mask_type == "input":
                in_features = round(50*(1-self.mask_ratio))*fps*duration
            else:
                in_features = 50*fps*duration
                
            # in_features = int(50*(1-self.mask_ratio))*fps*duration
            out_features = 128
            temporal_dim=768
            
            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)
                        
            model_path = '/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)
            # self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            
            self.proj_sync = nn.Linear(in_features=240, out_features=out_features)
            if self.sync_type=='add':
                self.sync_weight = nn.Parameter(torch.tensor(0.0))
            elif self.sync_type=="cross-attention":
                cross_attention_num_heads = 3 # MultiHeadCrossAttention
                self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, cross_attention_num_heads)
                
        
            self.in_features = in_features
            self.out_features = out_features

            self.SA_type = 'temporal_SA' # or 'temporal_SA'
            # self.Spa_Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            # self.Spa_Temp_pos_embedding = nn.Parameter(torch.randn(1, 50*10*10, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            if self.SA_type=='temporal_SA':        
                self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration*fps, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            elif self.SA_type=='spatial_SA':
                self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]

            # 获取 CLIP 的标准均值和标准差
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            # 手动在 GPU 上定义预处理管道
            self.preprocess_CLIP = transforms.Compose([
                # transforms.Resize((224, 224)),  # 调整大小
                transforms.Normalize(mean=clip_mean, std=clip_std)  # 归一化
            ])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0  # 将像素值缩放到 [0, 1]
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def forward(self, Video_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        Video_tensors = Video_list
        # video_sync_frames = [item["video_sync_frames"] for item in Video_list]
        # video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)
        
        original_videos = torch.cat(Video_tensors, dim=0)
        batch_size, time_length, _, _, _ = original_videos.size()
        # 记录哪些样本全为0，得到一个布尔张量 shape: (B,)
        is_zero = torch.all(original_videos == 0, dim=(1,2,3,4))
        # 后续继续使用 original_videos 作为视频数据
        Video_tensors = original_videos
        
        # 创建一个全零矩阵，形状与 video_hidden 相同
        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')


        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)

        if self.clip_model_name=='clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            if self.SA_type=='temporal_SA':
                # temporal SA
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h',b=batch_size,t=time_length) # [150, 100, 768]  
                video_hidden += self.Temp_pos_embedding
                video_hidden = self.Temp_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h',b=batch_size,t=time_length)
            elif self.SA_type=='spatial_SA':
                # spacial SA
                video_hidden += self.Spatial_pos_embedding
                video_hidden = self.Spatial_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length)

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1).to(device)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden) 
        # print('with sync: empty_visual_feat.shape: ', empty_visual_feat.shape)               
        return video_hidden, torch.ones(video_hidden.shape[0], video_hidden.shape[1]).to(device)


class VideoMAEWithSyncWithoutEmptyFeatureConditioner(Conditioner):
    CLIP_MODELS = ["video-mae-base"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "video-mae-base",
            video_fps: int = 5,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000, # 10*10*50 [t, fps, head]
            # in_features: int = 4700, # 47*2*50 [t, fps, head]
            project_out: bool = False,
            mask_ratio: float = 0.0,
            mask_type: str = "input",
            sync_type: str = "add"
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        # sa_depth=8        
        
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        # fps = 20
        print(f"video_fps: {video_fps}")
        fps = video_fps
        self.clip_model_name='video-mae-base'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.sync_type = sync_type
        
        if self.clip_model_name=='video-mae-base':
            if self.mask_type == "input":
                in_features = round(50*(1-self.mask_ratio))*fps*duration
            else:
                in_features = 50*fps*duration
            
            
            from transformers import AutoImageProcessor, VideoMAEModel
            self.processor = AutoImageProcessor.from_pretrained("/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--MCG-NJU--videomae-base/snapshots/dc740ceda42fce44faed2ea03c6d447db72f6af9")
            self.visual_encoder_model = VideoMAEModel.from_pretrained("/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--MCG-NJU--videomae-base/snapshots/dc740ceda42fce44faed2ea03c6d447db72f6af9")


            # in_features = int(50*(1-self.mask_ratio))*fps*duration
            out_features = 128
            temporal_dim=768
            
            self.empty_visual_feat = nn.Parameter(torch.zeros(1, 1568, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)
                        
            # model_path = '/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
            # self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)


            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            
            self.proj_sync = nn.Linear(in_features=240, out_features=out_features)
            if self.sync_type=='add':
                self.sync_weight = nn.Parameter(torch.tensor(0.0))
            elif self.sync_type=="cross-attention":
                cross_attention_num_heads = 3 # MultiHeadCrossAttention
                self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, cross_attention_num_heads)
                
        
            self.in_features = in_features
            self.out_features = out_features

            self.SA_type = 'temporal_SA' # or 'temporal_SA'
            # self.Spa_Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            # self.Spa_Temp_pos_embedding = nn.Parameter(torch.randn(1, 50*10*10, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            if self.SA_type=='temporal_SA':        
                self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Temp_pos_embedding = nn.Parameter(torch.randn(1, 1568, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]
            elif self.SA_type=='spatial_SA':
                self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
                self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim)) # spatial transformer:[1, 50*t*fps, 768]


    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def forward(self, Video_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        Video_tensors = Video_list
        # video_sync_frames = [item["video_sync_frames"] for item in Video_list]
        # video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)
        
        original_videos = torch.cat(Video_tensors, dim=0)  # [B, T, C, H, W]
        batch_size, time_length, _, _, _ = original_videos.size()
        # 记录哪些样本全为0，得到一个布尔张量 shape: (B,)
        is_zero = torch.all(original_videos == 0, dim=(1,2,3,4))
        # 后续继续使用 original_videos 作为视频数据 [B, T, C, H, W]
        Video_tensors = original_videos

        num_frames = 16
        pixel_values_list = []
        for b in range(batch_size):
            v = Video_tensors[b]  # [T, C, H, W]

            # 均匀采样 num_frames 帧（不足则重复最后一帧）
            if time_length <= 0:
                idx = torch.zeros(num_frames, dtype=torch.long)
            elif time_length == 1:
                idx = torch.zeros(num_frames, dtype=torch.long)
            else:
                idx = torch.linspace(0, time_length - 1, steps=num_frames).round().long()

            frames = v.index_select(0, idx)  # [num_frames, C, H, W]
            frames = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [num_frames, H, W, C]

            inputs_one = self.processor(list(frames), return_tensors="pt")
            pixel_values_list.append(inputs_one["pixel_values"])

        inputs = {"pixel_values": torch.cat(pixel_values_list, dim=0).to(device)}
        if self.clip_model_name=='video-mae-base':
            with torch.no_grad():
                outputs = visual_encoder_model(**inputs)
            video_hidden = outputs.last_hidden_state

            if self.SA_type=='temporal_SA':
                # temporal SA
                # video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h',b=batch_size,t=time_length) # [150, 100, 768]  
                video_hidden += self.Temp_pos_embedding
                video_hidden = self.Temp_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                # video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h',b=batch_size,t=time_length)
            elif self.SA_type=='spatial_SA':
                # spacial SA
                video_hidden += self.Spatial_pos_embedding
                video_hidden = self.Spatial_transformer(video_hidden)     # [B*t, head, 768] # [150, 100, 768]
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h',b=batch_size,t=time_length)

        # video_hidden = proj(video_hidden.view(-1, self.in_features))
        # video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1).to(device)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)
        # print('with sync: empty_visual_feat.shape: ', empty_visual_feat.shape)               
        return video_hidden, torch.ones(video_hidden.shape[0], video_hidden.shape[1]).to(device)


from transformers import AutoTokenizer, CLIPModel
class CLIPTextEncoderConditioner(Conditioner):
    def __init__(self, output_dim: int, clip_model_name: str = "clip-vit-base-patch32", project_out: bool = False):
        clip_output_dim = 512 
        super().__init__(clip_output_dim, output_dim, project_out=project_out)

        # 使用本地缓存路径
        clip_model_name = "/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--openai--clip-vit-base-patch32/text_encoder/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        # if clip_model_name == "clip-vit-base-patch32" or clip_model_name == "openai/clip-vit-base-patch32":
        #     clip_model_name = local_clip_path
        
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)

        # ** 关键补充：冻结CLIP模型的所有参数 **
        self.clip_model.eval() # 设置为评估模式
        for param in self.clip_model.parameters():
            param.requires_grad = False


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # 判断每个text的token个数，大于75就把后面的截断
        self.clip_model.eval().to(device)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        text_features = self.proj_out(text_features.unsqueeze(1))

        return text_features, torch.ones(text_features.shape[0], 1).to(device)


class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            mask_ratio: float = 0,
            mask_type: str = "input"
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad
        self.mask_ratio = mask_ratio
        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                t5_model_name = "/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"                
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model

    def random_masking_text(self, text, mask_ratio):
        """
        当text的单词数量*mask_ratio大于1时，随机mask掉text中的单词，分三种情况：
        1. In 80% of the cases, the masked tokens are replaced by self.tokenizer.unk_token.
        2. In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace (self.tokenizer.vocab中随机选择).
        3. In the 10% remaining cases, the masked tokens are left as is.
        """
        if text.endswith('<infer>'):
            text = text.replace('<infer>', '').strip()            
            return text
        
        words = text.split()  # 将文本分割成单词
        num_words = len(words)
        num_to_mask = int(num_words * mask_ratio)

        if num_to_mask < 2:
            return text  # 如果没有单词需要mask，返回原文本

        # 随机选择要mask的单词索引
        mask_indices = random.sample(range(num_words), num_to_mask)

        for idx in mask_indices:
            words[idx] = self.tokenizer.unk_token

        return ' '.join(words)  # 将单词列表重新组合成文本
        
    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)


        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # print(f'text sample: {texts[0]}')

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    
            
        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask    


class T5Conditioner_384(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            mask_ratio: float = 0,
            mask_type: str = "input"
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad
        self.mask_ratio = mask_ratio
        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        self.proj_seq_len_384 = nn.Linear(in_features=128, out_features=384)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                t5_model_name = "/mnt/shanghai2cephs/zeyuetian/project/ckpts/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"                
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model

    def random_masking_text(self, text, mask_ratio):
        """
        当text的单词数量*mask_ratio大于1时，随机mask掉text中的单词，分三种情况：
        1. In 80% of the cases, the masked tokens are replaced by self.tokenizer.unk_token.
        2. In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace (self.tokenizer.vocab中随机选择).
        3. In the 10% remaining cases, the masked tokens are left as is.
        """
        if text.endswith('<infer>'):
            text = text.replace('<infer>', '').strip()            
            return text
        
        words = text.split()  # 将文本分割成单词
        num_words = len(words)
        num_to_mask = int(num_words * mask_ratio)

        if num_to_mask < 2:
            return text  # 如果没有单词需要mask，返回原文本

        # 随机选择要mask的单词索引
        mask_indices = random.sample(range(num_words), num_to_mask)

        for idx in mask_indices:
            words[idx] = self.tokenizer.unk_token

        return ' '.join(words)  # 将单词列表重新组合成文本
        
    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)


        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    
            
        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        embeddings = self.proj_seq_len_384(embeddings.transpose(1, 2)).transpose(1, 2)
        return embeddings, attention_mask    


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
    # length = length if isinstance(length, int) else length.max()
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
import torch.nn.functional as F

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
            # self.empty_speech_feat = nn.Parameter(torch.zeros(1, 128, 768), requires_grad=True)
            nn.init.constant_(self.empty_speech_feat, 0)  


            self.norm_features = RMSNorm(768)
            self.norm = RMSNorm(768)
            # self.norm = nn.LayerNorm(768)


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


        # 记录哪些文本是 None
        is_none = [text is None for text in texts]
        # 如果全部都是 None，直接返回 empty_speech_feat
        if all(is_none):
            empty_speech_feat = self.empty_speech_feat.expand(batch_size, -1, -1).to(device)
            return empty_speech_feat, torch.ones(empty_speech_feat.shape[0], empty_speech_feat.shape[1]).to(device)
        
        # 将 None 替换为空字符串用于处理
        texts_processed = [text if text is not None else "" for text in texts]
        
        final_text_list = self.convert_char_to_pinyin(texts_processed)
        
        if isinstance(final_text_list, list):
            if self.tts_model_name == "tts-f5":
                text = self.list_str_to_idx(final_text_list, self.vocab_char_map).to(device)

        if text.dtype != torch.long:
            text = text.long()
        
        text_embed = self.text_embed(text, self.seq_len, drop_text=False)
        text_embed = self.norm_features(text_embed)

        # text_embed = self.norm(text_embed)
        # text_embed = self.proj_features(text_embed.to(torch.bfloat16))
        text_embed = self.proj_features(text_embed)
        
        text_embed = self.proj_seq_len(text_embed.transpose(1, 2)).transpose(1, 2)
        # 对于 None 的文本，使用 empty_speech_feat 替换
        empty_speech_feat = self.empty_speech_feat.expand(batch_size, -1, -1).to(device)
        is_none_tensor = torch.tensor(is_none, device=device).view(batch_size, 1, 1)
        text_embed = torch.where(is_none_tensor, empty_speech_feat, text_embed)
        
        text_embed = self.norm(text_embed)
        return text_embed, torch.ones(text_embed.shape[0], text_embed.shape[1]).to(device)


class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)
  
class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.cat(audio, dim=0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        
        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]
    

import torchaudio
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
        # self.RMSNorm = RMSNorm(768)
        # self.norm = nn.LayerNorm(768)
        self.norm = RMSNorm(768)
        
    def resample_and_pad(self, wavs, device):
        # 重采样音频tensor
        wavs = [wav.to(device).float() for wav in wavs]
        
        # 找到最长的长度
        max_length = max([wav.shape[-1] for wav in wavs])
        
        # 填充音频tensor到最长长度
        padded_wavs = [torch.nn.functional.pad(wav, (0, max_length - wav.shape[-1])) for wav in wavs]
        # 堆叠填充后的音频tensor
        stacked_wavs = torch.stack(padded_wavs)
        # 重采样整个批次的音频tensor
        resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sample_rate).to(device)
        resampled_wavs = resampler(stacked_wavs)
        return resampled_wavs

    def mask_mel_spectrogram(self, mels: torch.Tensor, task_tts_list: tp.List[bool] = None) -> torch.Tensor:
        """
        对mel频谱图进行随机mask
        Args:
            mels: shape为[b, 100, t]的tensor
            task_tts_list: 每个样本是否为TTS任务的列表，只有True的样本会被mask
        Returns:
            masked_mels: 被mask后的mel频谱图
        """
        batch_size, n_mels, seq_len = mels.shape
        device = mels.device
        
        # 判断模型是否可训练
        model_trainable = any(p.requires_grad for p in self.parameters())
        if model_trainable:
            # 为每个样本随机生成mask比例
            mask_ratios = torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        else:
            mask_ratios = torch.zeros(batch_size, device=device)
        
        # 为每个样本生成mask
        masked_mels = mels.clone()
        for i in range(batch_size):
            # 只对TASK_TTS为True的样本进行mask
            if task_tts_list[i]:
                # 计算需要mask的长度
                mask_len = int(seq_len * mask_ratios[i])
                # 随机选择起始位置
                start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
                # 创建mask
                masked_mels[i, :, start_pos:start_pos + mask_len] = 0
            
        return masked_mels, mask_ratios

    def forward(self, audio_input_prompts: tp.List[dict], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        # return self.empty_audio_feat, torch.ones(self.empty_audio_feat.shape[0], 1).to(device)

        wavs = []
        TASK_TTS = []
        for audio_input_prompt in audio_input_prompts:
            wavs.append(audio_input_prompt['audio_input_wav'])
            TASK_TTS.append(audio_input_prompt['TASK_TTS'])
        
        self.proj_out.to(device)
        
        # wavs = self.resample_and_pad(wavs,device)
        wavs = torch.stack(wavs, dim=0).to(device).float()

        mels = self.extractor(
            waveform=wavs,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        # 对mel频谱图进行mask，只对TASK_TTS为True的样本进行mask
        if self.mask_ratio_start < self.mask_ratio_end:
            mels, mask_ratios = self.mask_mel_spectrogram(mels, TASK_TTS)

        mels = self.proj_sequence_features(mels)
        mels = mels.transpose(1, 2)
        mels_emb = self.proj_features(mels)

        mels_emb = self.norm(mels_emb)

        return mels_emb, torch.ones(mels_emb.shape[0], mels_emb.shape[1]).to(device) 
    

class AudioMelConditioner_128(Conditioner):

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
            in_features_dim: int = 1723,
            project_out: bool = False,            
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

        # self.proj_features_128 = nn.Linear(in_features=1723, out_features=128)
        # self.proj_features_128 = nn.Linear(in_features=1876, out_features=128)        
        self.proj_features_128 = nn.Linear(in_features=in_features_dim, out_features=128)        
        
        self.proj_features = nn.Linear(in_features=self.n_mel_channels, out_features=768)     

    def resample_and_pad(self, wavs, device):
        # 重采样音频tensor
        wavs = [wav.to(device).float() for wav in wavs]
        
        # 找到最长的长度
        max_length = max([wav.shape[-1] for wav in wavs])
        
        # 填充音频tensor到最长长度
        padded_wavs = [torch.nn.functional.pad(wav, (0, max_length - wav.shape[-1])) for wav in wavs]
        # 堆叠填充后的音频tensor
        stacked_wavs = torch.stack(padded_wavs)
        # 重采样整个批次的音频tensor
        resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sample_rate).to(device)
        resampled_wavs = resampler(stacked_wavs)
        return resampled_wavs

    def mask_mel_spectrogram(self, mels: torch.Tensor) -> torch.Tensor:
        """
        对mel频谱图进行随机mask
        Args:
            mels: shape为[b, 100, t]的tensor
        Returns:
            masked_mels: 被mask后的mel频谱图
        """
        batch_size, n_mels, seq_len = mels.shape
        device = mels.device
        
        # 为每个样本随机生成mask比例
        mask_ratios = torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        
        # 为每个样本生成mask
        masked_mels = mels.clone()
        for i in range(batch_size):
            # 计算需要mask的长度
            mask_len = int(seq_len * mask_ratios[i])
            # 随机选择起始位置
            start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
            # 创建mask
            masked_mels[i, :, start_pos:start_pos + mask_len] = 0
            
        return masked_mels, mask_ratios

    def forward(self, wavs: tp.List[torch.tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.proj_out.to(device)
        
        # wavs = self.resample_and_pad(wavs,device)
        wavs = torch.cat(wavs, dim=0).to(device).float()

        mels = self.extractor(
            waveform=wavs,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        
        # 对mel频谱图进行mask
        if self.mask_ratio_start < self.mask_ratio_end:
            mels, mask_ratios = self.mask_mel_spectrogram(mels)

        mels_128 = self.proj_features_128(mels)
        mels_128 = mels_128.transpose(1, 2)
        mels_emb = self.proj_features(mels_128)

        return mels_emb, torch.ones(mels_emb.shape[0], 1).to(device) 
    

class AudioAutoencoderConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """

    def __init__(self, pretransform: Pretransform, output_dim: int, latent_seq_len: int = 237, mask_ratio_start: float = 0, mask_ratio_end: float = 0):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform      
        self.latent_seq_len = latent_seq_len
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        # self.empty_audio_feat = nn.Parameter(torch.zeros(1, self.latent_seq_len, self.proj_out.out_features), requires_grad=True)
        # nn.init.constant_(self.empty_audio_feat, 0)
        self.proj_features_128 = nn.Linear(in_features=self.latent_seq_len, out_features=128)
        
    def mask_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        对audio进行随机mask
        Args:
            audio: shape为[b, channels, length]的tensor
        Returns:
            masked_audio: 被mask后的audio，mask的地方置为0
        """
        batch_size, channels, seq_len = audio.shape
        device = audio.device
        
        # 为每个样本随机生成mask比例
        mask_ratios = torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        
        # 为每个样本生成mask
        masked_audio = audio.clone()
        for i in range(batch_size):
            # 计算需要mask的长度
            mask_len = int(seq_len * mask_ratios[i])
            # 随机选择起始位置
            start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
            # 创建mask，将mask区域置为0
            masked_audio[i, :, start_pos:start_pos + mask_len] = 0
            
        return masked_audio, mask_ratios

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)
        bs=len(audio)
        max_len = max([a.shape[-1] for a in audio])
        for i in range(bs):
            audio[i] = audio[i].to(device)
            # 对每个音频进行padding，使其长度一致
            pad_len = max_len - audio[i].shape[-1]
            if pad_len > 0:
                # 假设音频shape为(通道数, 长度)
                audio[i] = torch.nn.functional.pad(audio[i], (0, pad_len))
            # audio[i] = audio[i].unsqueeze(0)
            
        audio = torch.cat(audio, dim=0)


        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        # 对audio进行mask
        if self.mask_ratio_start < self.mask_ratio_end:
            audio, mask_ratios = self.mask_audio(audio)

        latents = self.pretransform.encode(audio)

        latents = self.proj_features_128(latents)
        latents = latents.permute(0, 2, 1)
        latents = self.proj_out(latents)
    
        return latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)
    

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

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "t5-384":
            conditioners[id] = T5Conditioner_384(**conditioner_config)
        elif conditioner_type == "clip" or conditioner_type == "mask-clip":
            conditioners[id] = CLIPConditioner(**conditioner_config)
        elif conditioner_type == "clip-with-sync":
            conditioners[id] = CLIPWithSyncConditioner(**conditioner_config)  
        elif conditioner_type == "clip-with-sync-wo-empty-feat":
            conditioners[id] = CLIPWithSyncWithoutEmptyFeatureConditioner(**conditioner_config)     
        elif conditioner_type == "video-mae-with-sync-wo-empty-feat":
            conditioners[id] = VideoMAEWithSyncWithoutEmptyFeatureConditioner(**conditioner_config)     
        elif conditioner_type == "dino-v2-encoder":
            conditioners[id] = DINOv2EncoderConditioner(**conditioner_config)
        elif conditioner_type == "clip-text-encoder":
            conditioners[id] = CLIPTextEncoderConditioner(**conditioner_config)                
        elif conditioner_type == "qwen_omni":
            conditioners[id] = QwenOmniWithSyncWithEmptyFeatureConditioner(**conditioner_config)
        elif conditioner_type == "qwen_omni_metaquery":
            conditioners[id] = MetaQueryWithQwenOmniConditioner(**conditioner_config)
        elif conditioner_type == "tts-f5" or conditioner_type == "tts-f5-token-level":
            conditioners[id] = TTSConditioner(**conditioner_config)             
        elif conditioner_type == "sync_feature":
            conditioners[id] = SynchformerConditioner(**conditioner_config)
        elif conditioner_type == "mel_spec":
            conditioners[id] = AudioMelConditioner(**conditioner_config)                                 
        elif conditioner_type == "mel_spec-w-empty-feature":
            conditioners[id] = AudioMelConditioner(**conditioner_config)  
        elif conditioner_type == "mel_spec-w-empty-feature-128":
            conditioners[id] = AudioMelConditioner_128(**conditioner_config)              
        elif conditioner_type == "clip-with-sync-w-empty-feat":
            conditioners[id] = CLIPWithSyncWithEmptyFeatureConditioner(**conditioner_config)
        elif conditioner_type == "video_sync_feature":
            conditioners[id] = SynchformerConditioner(**conditioner_config)

        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
            
        elif conditioner_type == "audio_autoencoder" or conditioner_type == "audio_autoencoder-w-empty-feature":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))
            if conditioner_type == "audio_autoencoder":
                conditioners[id] = AudioAutoencoderConditioner(pretransform, **conditioner_config)
            elif conditioner_type == "audio_autoencoder-w-empty-feature":
                conditioners[id] = AudioAutoencoderWithEmptyFeatureConditioner(pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)