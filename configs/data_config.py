"""Dataset-agnostic entry point for argument parsing."""

from configs.aroi_config import parse_args as _parse_aroi
from configs.duke_config import parse_args as _parse_duke
from configs.hcms_config import parse_args as _parse_hcms


def parse_args(dataset):
    """Parse arguments for ``dataset`` (e.g. ``"aroi"``, ``"duke"`` or ``"hcms"``)."""
    kind = dataset.lower()
    if kind == "aroi":
        return _parse_aroi()
    if kind == "duke":
        return _parse_duke()
    if kind == "hcms":
        return _parse_hcms()
    raise ValueError(f"Unsupported dataset: {dataset!r}")
