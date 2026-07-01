"""VisA dataset metadata."""

from .base import ClassMetadata, DatasetMetadata

VISA_METADATA = DatasetMetadata(
    name="visa",
    description="VisA dataset - Visual anomaly detection dataset",
    classes={
        "candle": ClassMetadata(
            name="candle",
            specie_names=[],
            description="Candle class"
        ),
        "capsules": ClassMetadata(
            name="capsules",
            specie_names=[],
            description="Capsules class"
        ),
        "cashew": ClassMetadata(
            name="cashew",
            specie_names=[],
            description="Cashew class"
        ),
        "chewinggum": ClassMetadata(
            name="chewinggum",
            specie_names=[],
            description="Chewing gum class"
        ),
        "fryum": ClassMetadata(
            name="fryum",
            specie_names=[],
            description="Fryum class"
        ),
        "macaroni1": ClassMetadata(
            name="macaroni1",
            specie_names=[],
            description="Macaroni1 class"
        ),
        "macaroni2": ClassMetadata(
            name="macaroni2",
            specie_names=[],
            description="Macaroni2 class"
        ),
        "pcb1": ClassMetadata(
            name="pcb1",
            specie_names=[],
            description="PCB1 class"
        ),
        "pcb2": ClassMetadata(
            name="pcb2",
            specie_names=[],
            description="PCB2 class"
        ),
        "pcb3": ClassMetadata(
            name="pcb3",
            specie_names=[],
            description="PCB3 class"
        ),
        "pcb4": ClassMetadata(
            name="pcb4",
            specie_names=[],
            description="PCB4 class"
        ),
        "pipe_fryum": ClassMetadata(
            name="pipe_fryum",
            specie_names=[],
            description="Pipe fryum class"
        )
    }
)
