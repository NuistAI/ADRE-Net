"""Intra-column Vertical Self-Attention (IVSA).

IVSA models the vertical (axial / depth) relationships *within* each A-scan
column of an OCT B-scan. A B-scan is ``[B, C, H, W]`` where ``H`` is the axial
(depth) axis and ``W`` indexes the A-scan columns; each ``(b, w)`` slice of
shape ``[C, H]`` is one A-scan. IVSA reshapes the feature map to ``[B*W, H, C]``
and applies multi-head self-attention along ``H``.

Exposed tensors (``expose=True``)
---------------------------------
``AscanAttentionMap`` : ``[B, W, num_heads, H, H]``  (query-depth x key-depth)
``AscanLogits``       : ``[B, W, num_heads, H, H]``
"""

import torch
from torch import nn
from einops import rearrange


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()

        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class IntraColumnSelfAttention(nn.Module):
    """Multi-head self-attention along the axial direction of one A-scan.

    Input  : ``[B*W, H, C]``
    Output : ``[B*W, H, C]``
    """

    def __init__(
        self,
        dim_size,
        num_heads,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        use_pos_embed=True,
        max_positions=256,
    ):
        super().__init__()

        assert dim_size % num_heads == 0, (
            f"dim_size={dim_size} must be divisible by num_heads={num_heads}"
        )

        self.dim_size = dim_size
        self.num_heads = num_heads
        self.head_dim = dim_size // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_positions, dim_size))

        self.q_proj = nn.Linear(dim_size, dim_size, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_size, dim_size, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_size, dim_size, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_size, dim_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def transpose_for_scores(self, x):
        # [B*W, H, C] -> [B*W, heads, H, head_dim]
        B_, H, C = x.shape
        x = x.view(B_, H, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x, expose=False):
        if self.use_pos_embed:
            x = x + self.pos_embed[:, : x.shape[1], :]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = self.transpose_for_scores(q)
        kh = self.transpose_for_scores(k)
        vh = self.transpose_for_scores(v)

        # [B*W, heads, H, H]
        logits = qh.matmul(kh.transpose(-1, -2)) * self.scale
        attn = torch.softmax(logits, dim=-1)
        attn_for_value = self.attn_drop(attn)

        out = attn_for_value.matmul(vh)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], self.dim_size)

        out = self.proj(out)
        out = self.proj_drop(out)

        if expose:
            aux = {"attn": attn, "logits": logits, "q": qh, "k": kh}
            return out, aux

        return out


# Backward-compatible alias (previous code name).
AscanSelfAttention = IntraColumnSelfAttention


class IVSA(nn.Module):
    """Intra-column Vertical Self-Attention block (A-scan / depth attention).

    Input  : ``[B, C, H, W]``
    Output : ``[B, C, H, W]``

    Each of the ``W`` columns is treated as one A-scan and attends along ``H``.
    """

    def __init__(
        self,
        in_channels,
        inner_channels=None,
        num_heads=8,
        attention_dropout_rate=0.0,
        qkv_bias=True,
        use_gamma=True,
        use_pos_embed=False,
    ):
        super().__init__()

        inner_channels = inner_channels or in_channels

        assert inner_channels % num_heads == 0, (
            f"inner_channels={inner_channels} must be divisible by num_heads={num_heads}"
        )

        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.num_heads = num_heads

        if in_channels != inner_channels:
            self.input_proj = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
            self.output_proj = nn.Conv2d(inner_channels, in_channels, kernel_size=1, bias=False)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        self.norm1 = nn.LayerNorm(inner_channels, eps=1e-6)
        self.attn = IntraColumnSelfAttention(
            dim_size=inner_channels,
            num_heads=num_heads,
            attn_drop=attention_dropout_rate,
            qkv_bias=qkv_bias,
            use_pos_embed=use_pos_embed,
        )
        if use_gamma:
            self.gamma_attn = nn.Parameter(1e-4 * torch.ones(inner_channels))
            self.gamma_ffn = nn.Parameter(1e-4 * torch.ones(inner_channels))
        else:
            self.gamma_attn = None
            self.gamma_ffn = None

    def forward(self, inputs, expose=False):
        B, C, H, W = inputs.shape
        assert C == self.in_channels, f"input C={C}, expected in_channels={self.in_channels}"

        x = self.input_proj(inputs)                       # [B, inner_channels, H, W]
        tokens = rearrange(x, "b c h w -> (b w) h c")     # [B*W, H, inner_channels]
        residual = tokens

        if expose:
            attn_out, attn_aux = self.attn(self.norm1(tokens), expose=True)
        else:
            attn_out = self.attn(self.norm1(tokens), expose=False)
            attn_aux = None

        if self.gamma_attn is not None:
            tokens = residual + self.gamma_attn * attn_out
        else:
            tokens = residual + attn_out

        x = rearrange(tokens, "(b w) h c -> b c h w", b=B, w=W)
        x = self.output_proj(x)

        if expose:
            attn = rearrange(attn_aux["attn"], "(b w) n h1 h2 -> b w n h1 h2", b=B, w=W)
            logits = rearrange(attn_aux["logits"], "(b w) n h1 h2 -> b w n h1 h2", b=B, w=W)
            aux = {
                "AscanAttentionMap": attn,   # [B, W, num_heads, H, H]
                "AscanLogits": logits,       # [B, W, num_heads, H, H]
            }
            return x, aux

        return x
