# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

from rope import VisionRotaryEmbedding

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True, rope_size=0, rotate=0, truncate_percentage=0.0, max_freq_lang=1.0, min_freq_lang=0.0, same_theta=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        self.rope = VisionRotaryEmbedding(head_dim//2, rope_size, rotate=rotate, truncate_percentage=truncate_percentage, max_freq_lang=max_freq_lang, min_freq_lang=min_freq_lang, same_theta=same_theta) if rope_size else None

    def forward(self, x):
        B, N, C = x.shape

        if self.flash:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            if self.rope is not None:
                q, k, v = qkv.unbind(dim=2)
                q = torch.cat((q[:, :1], self.rope(q[:, 1:])), dim=1)
                k = torch.cat((k[:, :1], self.rope(k[:, 1:])), dim=1)
                qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)
        else:
            #TODO non-flash-attn with rope not finished
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, num_heads=-1):
        super().__init__()
        self.eps = eps
        self.num_heads = num_heads
        if num_heads != -1:
            assert dim % num_heads == 0
            self.head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.num_heads == -1:
            norm = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        else:
            B, L, D = x.shape
            norm = x.view(B, L, self.num_heads, self.head_dim).norm(2, dim=-1, keepdim=True)
            norm = norm.expand(B, L, self.num_heads, self.head_dim)
            norm = norm.reshape(B, L, D) / (self.head_dim ** 0.5)
        return self.scale * x / (norm + self.eps)

class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4, flash=True, rope_size=0, rotate=0, truncate_percentage=0.0, max_freq_lang=1.0, min_freq_lang=0.0, same_theta=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            flash=flash, rope_size=rope_size, rotate=rotate, truncate_percentage=truncate_percentage, max_freq_lang=max_freq_lang, min_freq_lang=min_freq_lang, same_theta=same_theta)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4, flash=True, rope_size=0, rotate=0, truncate_percentage=0.0, max_freq_lang=1.0, min_freq_lang=0.0, same_theta=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            flash=flash, rope_size=rope_size, rotate=rotate, truncate_percentage=truncate_percentage, max_freq_lang=max_freq_lang, min_freq_lang=min_freq_lang, same_theta=same_theta)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,
                flash=True,
                rope=False,
                rotate=0,
                truncate_percentage=0.0,
                max_freq_lang=1.0,
                min_freq_lang=0.0,
                same_theta=False,
                **kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                flash=flash, rope_size=img_size // patch_size if rope else 0, rotate=rotate, truncate_percentage=truncate_percentage,
                max_freq_lang=max_freq_lang, min_freq_lang=min_freq_lang, same_theta=same_theta)
            for i in range(depth)])
           
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

@register_model
def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

# -----------------------------------------
# Base with rms
@register_model
def deit_base_patch16_LS_rms(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms2h(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=2), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms3h(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=3), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms4h(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=4), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms6h(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms12h(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=12), block_layers=Layer_scale_init_Block, **kwargs)
    return model
# -----------------------------------------


# -----------------------------------------
# Depth 12, Dimension 1280
@register_model
def deit_base_patch16_LS_rms_d1280(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1280, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms2h_d1280(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1280, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=2), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms4h_d1280(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1280, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=4), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms10h_d1280(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1280, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=10), block_layers=Layer_scale_init_Block, **kwargs)
    return model
# -----------------------------------------


# -----------------------------------------
# Depth 12, Dimension 2048+
@register_model
def deit_base_patch16_LS_rms_d2048(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=2048, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms_d2560(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=2560, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

# ----------------------------------------


# -----------------------------------------
# Depth 6, Dimension 768
@register_model
def deit_depth6_patch16_LS_rms_d768(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_depth6_patch16_LS_rms_d1280(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1280, depth=6, num_heads=16, mlp_ratio=2.5, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model
# ----------------------------------------


# -----------------------------------------
# Depth 4, rope
@register_model
def deit_depth4_patch16_LS_rms_d4096(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=4096, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, rope=True, **kwargs)
    return model

@register_model
def deit_depth4_patch16_LS_rms16h_d4096(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=4096, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6, num_heads=16), block_layers=Layer_scale_init_Block, rope=True, **kwargs)
    return model

@register_model
def deit_depth4_patch16_LS_rms_d2048(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=2048, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, rope=True, **kwargs)
    return model

@register_model
def deit_depth4_patch16_LS_rms_d1024(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, rope=True, **kwargs)
    return model
# ----------------------------------------

# -----------------------------------------
# Base with rope
@register_model
def deit_base_patch16_LS_rms_rope(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, rope=True, **kwargs)
    return model
# -----------------------------------------

@register_model
def deit_base_patch16_LS_rms_swi(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=2.6667, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, Mlp_block=SwiGLU, **kwargs)
    return model

@register_model
def deit_base_patch16_LS_rms_swi_rope(img_size=224,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=2.6667, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Layer_scale_init_Block, Mlp_block=SwiGLU, rope=True, **kwargs)
    return model