"""Dataset loaders for ADRE-Net."""

from dataset.aroi import Dataset as AROIDataset
from dataset.duke import Dataset as DukeDataset
from dataset.hcms import Dataset as HCMSDataset

_DATASET_CLASSES = {
    "aroi": AROIDataset,
    "duke": DukeDataset,
    "hcms": HCMSDataset,
}


def get_dataset_class(dataset):
    """Return the ``Dataset`` class for ``dataset`` (``"aroi"`` / ``"duke"`` / ``"hcms"``)."""
    kind = dataset.lower()
    if kind not in _DATASET_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    return _DATASET_CLASSES[kind]
