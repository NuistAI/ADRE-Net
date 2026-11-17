"""ADRE-Net: Anatomy-guided Directional Relation Encoding Network.

Top-level model definition.

Encoder (MASC)
--------------
``StripUNetEncoder`` builds a 5-stage U-Net encoder from
``MultiScaleStripConvBlock`` (MASC) blocks. Each MASC block is a multi-branch
*asymmetric strip convolution* module: a 3x3 CBR followed by three parallel
depth-wise strip branches (``k x 1`` + ``1 x k``) whose outputs are fused by a
1x1 convolution and a residual connection. Down-sampling is performed by a
learnable strided 3x3 convolution (LDS) before each MASC block. The per-stage
strip kernels follow the paper:

    E1 = (7,5)/(11,7)/(19,11), E2 = (5,3)/(7,5)/(11,7),
    E3 = (3,3)/(5,3)/(7,5),    E4 = (3,3)/(5,3)/(7,5).

Decoder (DDAB)
--------------
``LearnableDecoder`` upsamples with bilinear interpolation and
inserts a ``DDAB`` (Dual-Directional Attention Block) at the bottleneck and
after each skip-connection fusion. ``DDAB`` runs IVSA (intra-column vertical
self-attention) and IHSA (inter-column horizontal self-attention) in parallel
and fuses them with the SCFG gate.

Output
------
``forward`` returns ``{"segment_output": logits, "aux": aux}`` where ``aux``
collects the attention maps/logits of every DDAB block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ddab import DDAB
except ImportError:  # pragma: no cover - running as a script
    from ddab import DDAB


class CBR(nn.Module):
    """Conv + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StripConvBranch(nn.Module):
    """One asymmetric strip-convolution branch: depthwise ``k x 1`` + ``1 x k``.

    The horizontal kernel is ``kernel_size`` and the vertical kernel is derived
    as ``max(3, kernel_size - (kernel_size // 4) * 2)``, giving the paper's
    ``(7,5)``, ``(11,7)``, ``(19,11)``, ... asymmetric pairs.
    """

    def __init__(self, channel, kernel_size):
        super().__init__()
        row_kernel = kernel_size
        row_pad = row_kernel // 2
        col_kernel = max(3, kernel_size - (kernel_size // 4) * 2)
        col_pad = col_kernel // 2

        self.branch = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(row_kernel, 1),
                      padding=(row_pad, 0), groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, col_kernel),
                      padding=(0, col_pad), groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.branch(x)


class MultiScaleStripConvBlock(nn.Module):
    """Multi-Scale Asymmetric Strip Convolution (MASC) block.

    CBR + three parallel strip branches (DW ``k x 1`` + DW ``1 x k``) fused by a
    1x1 convolution and a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(7, 11, 21),
                 fuse_kernel=1, fuse_padding=0, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.cbr = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.branches = nn.ModuleList([StripConvBranch(out_channels, k) for k in kernel_sizes])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(kernel_sizes), out_channels,
                      kernel_size=fuse_kernel, padding=fuse_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res_proj = nn.Identity() if use_residual else None

    def forward(self, x):
        x = self.cbr(x)
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.fuse(out)
        if self.use_residual:
            out = out + x
        return out


# Paper acronym alias.
MASC = MultiScaleStripConvBlock


class Encoder(nn.Module):
    """Simple VGG-style encoder (kept for reference / ablation)."""

    def __init__(self, channels, in_channels=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.block = VGGBlock
        self.layers.append(self.block(in_channels, channels[0]))
        for idx in range(len(channels) - 1):
            self.layers.append(nn.Conv2d(channels[idx], channels[idx],
                                         kernel_size=3, stride=2, padding=1, bias=False))
            self.layers.append(self.block(channels[idx], channels[idx + 1]))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if not isinstance(layer, nn.Conv2d):
                features.append(x)
        return features


class DownBlock(nn.Module):
    """Learnable down-sampling (LDS) + MASC block."""

    def __init__(self, in_channels, out_channels, kernel_sizes=(7, 11, 21),
                 fuse_kernel=1, fuse_padding=0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.block = MultiScaleStripConvBlock(in_channels, out_channels, kernel_sizes=kernel_sizes)

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Decoder(nn.Module):
    """Plain bilinear decoder (kept for reference / ablation)."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList()
        self.block = VGGBlock
        for idx in range(len(channels) - 2, -1, -1):
            self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.layers.append(self.block(channels[idx] + channels[idx + 1], channels[idx]))

    def forward(self, features):
        x = features[-1]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Upsample):
                x = layer(x)
                x = torch.cat([x, features[-(i // 2 + 2)]], dim=1)
            else:
                x = layer(x)
        return x


class LearnableDecoder(nn.Module):
    """Decoder with bilinear up-sampling and a DDAB after each fusion stage."""

    def __init__(self, channels):
        super().__init__()
        self.bottleneck_cascade = DDAB(
            in_channels=channels[-1],
            inner_channels=channels[-1],
            num_heads=8,
            use_pos_embed=False,
        )
        self.layers = nn.ModuleList()
        self.block = VGGBlock
        for idx in range(len(channels) - 2, -1, -1):
            self.layers.append(self.block(channels[idx] + channels[idx + 1], channels[idx]))
            self.layers.append(DDAB(
                in_channels=channels[idx],
                inner_channels=channels[idx],
                num_heads=8,
            ))

    @staticmethod
    def _merge_aux(target, source):
        for k, v in source.items():
            target.setdefault(k, []).append(v)

    def forward(self, features):
        aux = {}
        x, aux_n = self.bottleneck_cascade(features[-1], expose=True)
        self._merge_aux(aux, aux_n)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DDAB):
                x, aux_i = layer(x, expose=True)
                self._merge_aux(aux, aux_i)
            else:
                x = F.interpolate(x, size=features[-(i // 2 + 2)].shape[2:],
                                  mode="bilinear", align_corners=False)
                x = torch.cat([x, features[-(i // 2 + 2)]], dim=1)
                x = layer(x)
        return x, aux


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4).flatten(0, 1)
        x = self.conv2d(x)
        return x


class StripUNetEncoder(nn.Module):
    """MASC encoder. Outputs five multi-scale features f1..f5 (H, H/2, ..., H/16)."""

    def __init__(
        self,
        in_channels=1,
        channels=(64, 128, 256, 512, 1024),
        kernel_sizes=((7, 11, 19), (5, 7, 11), (3, 5, 7), (3, 5, 7)),
    ):
        super().__init__()
        if isinstance(kernel_sizes[0], int):
            kernel_sizes = (kernel_sizes,) * len(channels)

        self.stem = MultiScaleStripConvBlock(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_sizes=kernel_sizes[0],
        )
        self.down1 = DownBlock(channels[0], channels[1], kernel_sizes=kernel_sizes[1])
        self.down2 = DownBlock(channels[1], channels[2], kernel_sizes=kernel_sizes[2])
        self.down3 = DownBlock(channels[2], channels[3], kernel_sizes=kernel_sizes[3])
        self.down4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.stem(x)     # H, W
        f2 = self.down1(f1)   # H/2
        f3 = self.down2(f2)   # H/4
        f4 = self.down3(f3)   # H/8
        f5 = self.down4(f4)   # H/16
        f5 = self.conv4(f5)
        return [f1, f2, f3, f4, f5]


class ADRE_Net(nn.Module):
    """ADRE-Net for joint segmentation of retinal layers and fluids in OCT.

    Parameters
    ----------
    num_classes : int
        Number of output classes (8 for AROI).
    channels : sequence[int]
        Per-stage channel widths (default ``[32, 64, 128, 256, 512]``).
    """

    def __init__(self, num_classes, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.encoder = StripUNetEncoder(
            in_channels=3,
            channels=channels,
            kernel_sizes=((7, 11, 19), (5, 7, 11), (3, 5, 7), (3, 5, 7)),
        )
        self.decoder = LearnableDecoder(channels)
        self.segmentation_head = SegmentationHead(
            in_channels=channels[0],
            out_channels=num_classes,
        )

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputimg = inputs["image"]
        elif isinstance(inputs, torch.Tensor):
            inputimg = inputs
        if inputimg.dim() == 4 and inputimg.size(1) == 1:
            inputimg = inputimg.repeat(1, 3, 1, 1)
        features = self.encoder(inputimg)
        features, aux = self.decoder(features)
        logits = self.segmentation_head(features)
        return {"segment_output": logits, "aux": aux}
