from math import pi
import torch.nn.functional as F
import math

import torch
from torch import nn

from einops import rearrange, repeat
import numpy as np
import pdb  # 添加调试器


def broadcat(freqss, dim = -1):
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    assert len(shape_lens) == 1, 'freqss must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=14,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        rotate = 0,
        truncate_percentage = 0.0,
        max_freq_lang = 1.0,
        min_freq_lang = 0.0,
        same_theta = True,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # freqs = max_freq_lang / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            if rotate == 0:
                freqs = max_freq_lang / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            else:
                if same_theta:
                    effective_dim = dim // rotate
                else:
                    effective_dim = dim
                # freqs = max_freq_lang / (theta ** (torch.arange(0, effective_dim, 2)[:(effective_dim // 2)].float() / effective_dim))
                if min_freq_lang < 1e-6:
                    min_freq_lang = max_freq_lang / (theta ** ((effective_dim - 2) / effective_dim))

                dim_half = effective_dim // 2
                freqs = min_freq_lang * (max_freq_lang/min_freq_lang) ** (torch.arange(dim_half-1, -1, -1).float() / (dim_half-1))
                if truncate_percentage > 0:
                    assert len(freqs.shape) == 1, "freqs must be 1D"
                    num_to_truncate = int(freqs.shape[0] * truncate_percentage)
                    freqs[-num_to_truncate:] = 0
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.pt_seq_len=pt_seq_len
        self.register_buffer("freqs", freqs)
        self.rotate = rotate
        self.same_theta = same_theta

    def forward(self, x): 
        ft_seq_len = int(np.sqrt(x.shape[1])) # if x.shape[1] = 196 (14*14 patches), then ft_seq_len = 14
        t = torch.arange(ft_seq_len, device=x.device, dtype=x.dtype) / ft_seq_len * self.pt_seq_len # if ft_seq_len = pt_seq_len = 14, then t = [0, 1, 2, ..., 13]

        freqs = torch.einsum('..., f -> ... f', t, self.freqs)

        # Original implementation
        # 调试断点2: 查看频率矩阵
        # 可用的变量: self.freqs, freqs
        # pdb.set_trace()  # 断点2: 检查初始频率矩阵
        # breakpoint()
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2) # 14*32
        if self.same_theta:
            assert freqs.shape[1] % 4 == 0
            # breakpoint()
            freqs_grouped = freqs.reshape(ft_seq_len, freqs.shape[1] // 4, 4)
            freqs_grouped = freqs_grouped.repeat_interleave(self.rotate, dim=1) # 由于后面分割的时候是四个一组四个一组分割，所以提前重复好了，使得未来分割出来每个方向的freq都相同
            freqs = freqs_grouped.reshape(ft_seq_len, -1)

        # 调试断点3: 查看重复后的频率
        # 可用的变量: freqs (重复后的)
        # pdb.set_trace()  # 断点3: 检查重复后的频率
        # breakpoint()
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1) # 14*14*64

            # 调试断点4: 查看2D广播后的频率
            # 可用的变量: freqs (2D广播后的)
            # pdb.set_trace()  # 断点4: 检查2D广播后的频率

        if self.rotate > 0:
            # if self.rotate = 1, then freqs is unchanged
            assert x.shape[-1] % (self.rotate * 4) == 0
            freqs = freqs.view(ft_seq_len, ft_seq_len, -1, self.rotate * 4) # if self.rotate = 2, then freqs.shape = (14, 14, 8, 8), note that before reshaping, the last dimension is 64, with first 32 for x axis and the second 32 for y axis (y axis is rotating 90 degrees, and we rotate by 0< angle < 90 degrees for both x and y axis)
            for i in range(1, self.rotate):
                freqs[..., i * 4:(i + 1) * 4] = rotate_freqs(freqs[..., i * 4:(i + 1) * 4], i * 90 / self.rotate) # if self.rotate = 2, then angle_deg = 45
            freqs = freqs.view(ft_seq_len, ft_seq_len, -1)

        freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1])
        return  x * freqs_cos + rotate_half(x) * freqs_sin

# def rotate_freqs(freqs, angle_deg):
#     assert freqs.ndim == 3 and freqs.shape[0] == freqs.shape[1], "Input must have shape (n, n, d)"
#     n, _, d = freqs.shape
#     angle_rad = math.radians(angle_deg)

#     # Change shape from [H, W, C] to [1, C, H, W] for grid_sample
#     freqs = freqs.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, C, H, W]

#     # Build affine rotation matrix
#     theta = torch.tensor([
#         [ math.cos(angle_rad), -math.sin(angle_rad), 0.0],
#         [ math.sin(angle_rad),  math.cos(angle_rad), 0.0]
#     ], dtype=torch.float32).unsqueeze(0)

#     # Generate a sampling grid
#     grid = F.affine_grid(theta, freqs.size(), align_corners=True)

#     # Apply the same rotation to all channels using bilinear interpolation
#     # Use 'border' to fill missing values with edge values
#     rotated = F.grid_sample(freqs, grid, mode='bilinear', padding_mode='border', align_corners=True)

#     # Convert back from [1, C, H, W] to [H, W, C]
#     return rotated.squeeze(0).permute(1, 2, 0)

def rotate_freqs(freqs, angle_deg):
    # if self.rotate = 1, then freqs.shape = (14, 14, 16, 4)
    # breakpoint()
    assert freqs.ndim == 4 and freqs.shape[0] == freqs.shape[1], "Input must have shape (n, n, d1, d2)"
    n, _, d1, d2 = freqs.shape
    freq_type = freqs.dtype
    angle_rad = math.radians(angle_deg)

    # Reshape from (n, n, d1, d2) → (n, n, d1 * d2)
    freqs = freqs.reshape(n, n, -1)

    # Permute to (1, C, H, W) where C = d1 * d2
    freqs = freqs.permute(2, 0, 1).unsqueeze(0)

    # Rotation matrix (2x3)
    theta = torch.tensor([
        [ math.cos(angle_rad), -math.sin(angle_rad), 0.0],
        [ math.sin(angle_rad),  math.cos(angle_rad), 0.0]
    ], dtype=torch.float32, device=freqs.device).unsqueeze(0)

    freqs = freqs.to(torch.float32)

    # Build sampling grid
    grid = F.affine_grid(theta, freqs.size(), align_corners=True)

    # Rotate using bilinear interpolation, with border padding
    rotated = F.grid_sample(freqs, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Convert back: (1, C, H, W) → (H, W, C)
    rotated = rotated.squeeze(0).permute(1, 2, 0).to(freq_type)

    # Reshape back to (n, n, d1, d2)
    return rotated.reshape(n, n, d1, d2)