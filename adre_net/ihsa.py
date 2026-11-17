"""Inter-column Horizontal Self-Attention (IHSA).

IHSA models the lateral (horizontal) relationships *between* A-scan columns of
an OCT B-scan. Each column is summarized into a token by max/average-pooling
along the axial (depth) axis, and self-attention is computed over the ``W``
column tokens. The value stream is the *unpooled* feature map (via a 1x1 conv),
so the attention map is applied back to the full-resolution features.

Exposed tensors (``expose=True``)
---------------------------------
``BscanAttentionMap`` : ``[B, num_heads, W, W]``  (inter-column)
``BscanLogits``       : ``[B, num_heads, W, W]``
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class IHSA(nn.Module):
    """Inter-column Horizontal Self-Attention block (B-scan / column attention)."""

    def __init__(
        self,
        in_channels,
        inner_channels=32,
        num_heads=8,
        dropout_rate=0.0,
        use_gamma=False,
        use_pos_embed=False,
        max_positions=256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.num_heads = num_heads
        assert inner_channels % num_heads == 0, (
            f"channels={inner_channels} must be divisible by num_heads={num_heads}"
        )

        self.v_head_channels = inner_channels // num_heads

        C = inner_channels
        self.qk_in_dim = 2 * C
        qk_dim = self.qk_in_dim
        assert qk_dim % num_heads == 0, (
            f"qk_dim={qk_dim} must be divisible by num_heads={num_heads}"
        )
        self.head_dim = qk_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_positions, self.qk_in_dim))

        if in_channels != inner_channels:
            self.input_proj = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
            self.output_proj = nn.Conv2d(inner_channels, in_channels, kernel_size=1, bias=False)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        self.q_proj = nn.Linear(self.qk_in_dim, qk_dim)
        self.k_proj = nn.Linear(self.qk_in_dim, qk_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        self.v_proj = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(C, C, kernel_size=1, bias=False)

        if use_gamma:
            self.gamma_attn = nn.Parameter(1e-2 * torch.ones(1))
        else:
            self.gamma_attn = None

    def _build_qk_tokens(self, x):
        """Pool along the axial axis to build column tokens: ``[B, W, 2C]``."""
        avg_h = x.mean(dim=2)               # [B, C, W]
        max_h = x.amax(dim=2)               # [B, C, W]
        pooled = torch.cat([avg_h, max_h], dim=1)  # [B, 2C, W]
        return rearrange(pooled, "b c s -> b s c")  # [B, W, 2C]

    def _transpose_for_scores(self, x):
        B, S, D = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, inputs, expose=False):
        x = self.input_proj(inputs)         # [B, inner_channels, H, W]
        B, C, H, W = x.shape
        assert C == self.inner_channels, (
            f"projected C={C}, expected inner_channels={self.inner_channels}"
        )
        identity = x

        qk_tokens = self._build_qk_tokens(x)   # [B, W, 2C]
        if self.use_pos_embed:
            qk_tokens = qk_tokens + self.pos_embed[:, : qk_tokens.shape[1], :]

        q = self.q_proj(qk_tokens)             # [B, W, qk_dim]
        k = self.k_proj(qk_tokens)

        qh = self._transpose_for_scores(q)     # [B, heads, W, head_dim]
        kh = self._transpose_for_scores(k)

        logits = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        attn = F.softmax(logits, dim=-1)       # [B, heads, W, W]

        v_feat = self.v_proj(x)                # [B, C, H, W]
        h = self.num_heads
        C_h = C // h
        v_tokens = v_feat.view(B, h, C_h, H, W)

        out = torch.einsum("bhij,bhcyj->bhcyi", attn, v_tokens)
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        if self.gamma_attn is not None:
            out = identity + self.gamma_attn * out
        else:
            out = identity + out

        out = self.output_proj(out)

        if expose:
            aux = {
                "BscanAttentionMap": attn,     # [B, heads, W, W]
                "BscanLogits": logits,         # [B, heads, W, W]
            }
            return out, aux

        return out
