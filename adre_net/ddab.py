"""Dual-Directional Attention Block (DDAB) and Spatial-Channel Fusion Gate (SCFG).

DDAB jointly models the two anatomical directions of an OCT B-scan:

* :class:`IVSA` — intra-column vertical self-attention (A-scan / depth), and
* :class:`IHSA` — inter-column horizontal self-attention (B-scan / lateral).

The two directional branches run in parallel and are adaptively fused by the
:class:`SCFG` (spatial + channel gate): ``out = gate * x_ivsa + (1 - gate) * x_ihsa``.
"""

import torch
from torch import nn

try:
    from .ivsa import IVSA
    from .ihsa import IHSA
except ImportError:  # pragma: no cover - running as a script
    from ivsa import IVSA
    from ihsa import IHSA


class SCFG(nn.Module):
    """Spatial-Channel Fusion Gate.

    Fuses two same-shaped feature maps ``x1`` and ``x2`` with a learned gate
    combining a 7x7 spatial branch and a channel (GAP + MLP) branch::

        out = sigmoid(gs + gc) * x1 + (1 - sigmoid(gs + gc)) * x2
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=7, padding=3, bias=True),
        )
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x1, x2):
        feat = torch.cat([x1, x2], dim=1)
        gs = self.spatial_gate(feat)     # [B, 1, H, W]
        gc = self.channel_gate(feat)     # [B, C, 1, 1]
        gate = torch.sigmoid(gs + gc)
        out = gate * x1 + (1 - gate) * x2
        return out, gate


class DDAB(nn.Module):
    """Dual-Directional Attention Block (IVSA + IHSA fused by SCFG)."""

    def __init__(
        self,
        in_channels,
        inner_channels=384,
        num_heads=8,
        post_ffn_bottleneck_ratio=0.25,
        post_ffn_dropout_rate=0.0,
        use_pos_embed=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.inner_channels = inner_channels

        # Branch-aware modulation (AB = A-scan -> B-scan, BA = B-scan -> A-scan).
        self.ab_gamma = nn.Parameter(torch.ones(1, inner_channels, 1, 1))
        self.ab_beta = nn.Parameter(torch.zeros(1, inner_channels, 1, 1))
        self.ba_gamma = nn.Parameter(torch.ones(1, inner_channels, 1, 1))
        self.ba_beta = nn.Parameter(torch.zeros(1, inner_channels, 1, 1))

        if in_channels != inner_channels:
            self.input_proj = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
            self.output_proj = nn.Conv2d(inner_channels, in_channels, kernel_size=1, bias=False)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        self.ascan = IVSA(
            in_channels=inner_channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            use_gamma=True,
            use_pos_embed=use_pos_embed,
        )
        self.bscan = IHSA(
            in_channels=inner_channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            use_gamma=True,
            use_pos_embed=use_pos_embed,
        )
        self.fusion_gate = SCFG(channels=inner_channels, reduction=16)

    @staticmethod
    def _pick_main_aux(aux_a=None, aux_b=None):
        out = {}
        if aux_a is not None:
            if "AscanAttentionMap" in aux_a:
                out["AscanAttentionMap"] = aux_a["AscanAttentionMap"]
            if "AscanLogits" in aux_a:
                out["AscanLogits"] = aux_a["AscanLogits"]
        if aux_b is not None:
            if "BscanAttentionMap" in aux_b:
                out["BscanAttentionMap"] = aux_b["BscanAttentionMap"]
            if "BscanLogits" in aux_b:
                out["BscanLogits"] = aux_b["BscanLogits"]
        return out

    def forward(self, x, expose=False):
        x_proj = self.input_proj(x)
        aux = {}

        if expose:
            xa, aux_a = self.ascan(x_proj, expose=True)
            xb, aux_b = self.bscan(x_proj, expose=True)
            aux.update(self._pick_main_aux(aux_a, aux_b))
        else:
            xa = self.ascan(x_proj, expose=False)
            xb = self.bscan(x_proj, expose=False)

        out, _gate_map = self.fusion_gate(xa, xb)
        out = self.output_proj(out)

        if expose:
            return out, aux
        return out
