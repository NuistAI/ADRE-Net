"""ADRE-Net — Anatomy-guided Directional Relation Encoding Network.

Public API
----------
``ADRE_Net``
    The full joint segmentation network for retinal layers + fluids in OCT.
``choice_backbone``
    Backbone factory. Only ``"adre_net"`` is accepted.
"""

from adre_net.adre_net import ADRE_Net

__all__ = ["ADRE_Net", "choice_backbone"]


def choice_backbone(kind, num_classes, input_config=None, **kwargs):
    """Backbone factory (``kind`` is case-insensitive)."""
    kind = (kind or "adre_net").lower()
    if kind == "adre_net":
        return ADRE_Net(num_classes=num_classes)
    raise ValueError(f"Unknown backbone: {kind!r}. Only 'adre_net' is supported.")
