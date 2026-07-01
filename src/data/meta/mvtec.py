"""MVTec AD dataset metadata."""

from .base import ClassMetadata, DatasetMetadata

MVTEC_METADATA = DatasetMetadata(
    name="mvtec",
    description="MVTec AD dataset - Industrial anomaly detection dataset",
    classes={
        "bottle": ClassMetadata(
            name="bottle",
            specie_names=["good", "contamination", "broken_small", "broken_large"],
            description="Bottle class with contamination and breakage defects"
        ),
        "cable": ClassMetadata(
            name="cable",
            specie_names=["good", "poke_insulation", "missing_cable", "cut_outer_insulation", "cut_inner_insulation", "bent_wire", "missing_wire", "cable_swap", "combined"],
            description="Cable class with insulation and wire defects"
        ),
        "capsule": ClassMetadata(
            name="capsule",
            specie_names=["good", "scratch", "squeeze", "poke", "faulty_imprint", "crack"],
            description="Capsule class with surface and imprint defects"
        ),
        "carpet": ClassMetadata(
            name="carpet",
            specie_names=["good", "metal_contamination", "hole", "cut", "thread", "color"],
            description="Carpet class with contamination and damage defects"
        ),
        "grid": ClassMetadata(
            name="grid",
            specie_names=["good", "metal_contamination", "glue", "thread", "bent", "broken"],
            description="Grid class with contamination and structural defects"
        ),
        "hazelnut": ClassMetadata(
            name="hazelnut",
            specie_names=["good", "hole", "cut", "print", "crack"],
            description="Hazelnut class with surface defects"
        ),
        "leather": ClassMetadata(
            name="leather",
            specie_names=["good", "poke", "glue", "cut", "fold", "color"],
            description="Leather class with surface and structural defects"
        ),
        "metal_nut": ClassMetadata(
            name="metal_nut",
            specie_names=["good", "scratch", "flip", "color", "bent"],
            description="Metal nut class with surface and orientation defects"
        ),
        "pill": ClassMetadata(
            name="pill", 
            specie_names=["good", "scratch", "crack", "contamination", "pill_type", "color", "faulty_imprint", "combined"],
            description="Pill class with various manufacturing defects"
        ),
        "screw": ClassMetadata(
            name="screw",
            specie_names=["good", "thread_side", "scratch_head", "manipulated_front", "scratch_neck", "thread_top"],
            description="Screw class with threading and surface defects"
        ),
        "tile": ClassMetadata(
            name="tile",
            specie_names=["good", "rough", "oil", "crack", "gray_stroke", "glue_strip"],
            description="Tile class with surface and contamination defects"
        ),
        "toothbrush": ClassMetadata(
            name="toothbrush",
            specie_names=["good", "defective"],
            description="Toothbrush class with manufacturing defects"
        ),
        "transistor": ClassMetadata(
            name="transistor",
            specie_names=["good", "bent_lead", "misplaced", "cut_lead", "damaged_case"],
            description="Transistor class with lead and case defects"
        ),
        "wood": ClassMetadata(
            name="wood",
            specie_names=["good", "scratch", "hole", "liquid", "color", "combined"],
            description="Wood class with surface and contamination defects"
        ),
        "zipper": ClassMetadata(
            name="zipper",
            specie_names=["good", "rough", "split_teeth", "squeezed_teeth", "fabric_border", "broken_teeth", "fabric_interior", "combined"],
            description="Zipper class with mechanical defects"
        ),
    }
)
