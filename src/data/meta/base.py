"""Base classes and common definitions for dataset metadata."""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ClassMetadata:
    """Container for class metadata."""
    name: str
    specie_names: List[str] = field(default_factory=list)
    description: str = ""
    normal_label: str = "good"  # Normal label (e.g. 'ok' for BTAD)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def get_anomaly_species(self) -> List[str]:
        """Return anomaly specie_names, excluding normal_label."""
        return [s for s in self.specie_names if s != self.normal_label]
    
    def get_normal_species(self) -> List[str]:
        """Return the normal specie_name, if normal_label is present."""
        return [self.normal_label] if self.normal_label in self.specie_names else []
    
    def has_species_info(self) -> bool:
        """Return whether specie_name information is available."""
        return len(self.specie_names) > 0
    
    def get_all_species(self) -> List[str]:
        """Return all specie_names, including normal and anomaly species."""
        return self.specie_names.copy()


@dataclass
class DatasetMetadata:
    """Container for dataset metadata."""
    name: str
    classes: Dict[str, ClassMetadata]
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return list(self.classes.keys())
    
    def get_class_metadata(self, class_name: str) -> ClassMetadata:
        """Return metadata for a specific class."""
        if class_name not in self.classes:
            raise ValueError(f"Unknown class name: {class_name}")
        return self.classes[class_name]
    
    def has_class(self, class_name: str) -> bool:
        """Return whether the specified class exists."""
        return class_name in self.classes
    
    def get_classes_with_species_info(self) -> List[str]:
        """Return class names that include specie_name information."""
        return [name for name, meta in self.classes.items() if meta.has_species_info()]
    
    def get_all_specie_names(self) -> List[str]:
        """Return all specie_names across classes, with duplicates removed."""
        all_species = set()
        for class_meta in self.classes.values():
            all_species.update(class_meta.specie_names)
        return sorted(list(all_species))
