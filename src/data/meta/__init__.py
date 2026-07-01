"""Dataset metadata modules."""

from typing import Dict, List
from .base import ClassMetadata, DatasetMetadata
from .mvtec import MVTEC_METADATA
from .visa import VISA_METADATA

# Consolidated metadata for all datasets.
DATASET_METADATA: Dict[str, DatasetMetadata] = {
    "mvtec": MVTEC_METADATA,
    "visa": VISA_METADATA,
}

# Public API functions.
def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Return metadata for a dataset name."""
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return DATASET_METADATA[dataset_name]

def get_class_names(dataset_name: str) -> List[str]:
    """Return class names for a dataset name."""
    return get_dataset_metadata(dataset_name).get_class_names()

def get_specie_names(dataset_name: str, class_name: str) -> List[str]:
    """Return specie_names for a specific dataset/class pair."""
    dataset_meta = get_dataset_metadata(dataset_name)
    class_meta = dataset_meta.get_class_metadata(class_name)
    return class_meta.specie_names

def get_anomaly_species(dataset_name: str, class_name: str) -> List[str]:
    """Return anomaly specie_names for a specific dataset/class pair."""
    dataset_meta = get_dataset_metadata(dataset_name)
    class_meta = dataset_meta.get_class_metadata(class_name)
    return class_meta.get_anomaly_species()

def get_normal_species(dataset_name: str, class_name: str) -> List[str]:
    """Return normal specie_names for a specific dataset/class pair."""
    dataset_meta = get_dataset_metadata(dataset_name)
    class_meta = dataset_meta.get_class_metadata(class_name)
    return class_meta.get_normal_species()

def list_available_datasets() -> List[str]:
    """Return the list of available datasets."""
    return list(DATASET_METADATA.keys())

def has_species_info(dataset_name: str, class_name: str) -> bool:
    """Return whether a dataset/class pair has specie_name information."""
    dataset_meta = get_dataset_metadata(dataset_name)
    class_meta = dataset_meta.get_class_metadata(class_name)
    return class_meta.has_species_info()

def get_classes_with_species_info(dataset_name: str) -> List[str]:
    """Return class names with specie_name information for the specified dataset."""
    dataset_meta = get_dataset_metadata(dataset_name)
    return dataset_meta.get_classes_with_species_info()

# Exports for backward compatibility.
__all__ = [
    "DATASET_METADATA",
    "get_dataset_metadata",
    "get_class_names", 
    "get_specie_names",
    "get_anomaly_species",
    "get_normal_species",
    "list_available_datasets",
    "has_species_info",
    "get_classes_with_species_info",
    "ClassMetadata",
    "DatasetMetadata",
]
