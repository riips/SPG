from dataclasses import dataclass, field
from enum import Flag, auto
import logging
from typing import Any, Iterator, KeysView, ItemsView, List, Optional, Protocol, Tuple, TypedDict, TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from data import ImageRecord

log = logging.getLogger(__name__)


class ReturnFields(Flag):
    NONE = 0
    FEATURES = auto()
    SIMILARITY = auto()
    SCORE = auto()
    LOSS = auto()


@dataclass
class ScoreOptions:
    compute_score: Optional[bool] = None
    compute_map: Optional[bool] = None
    aggregate_map: Optional[bool] = None
    smooth_map: Optional[bool] = None
    sigma: Optional[float] = None
    temperature: Optional[float] = None


@dataclass
class ForwardOptions:
    returns: ReturnFields
    score_options: ScoreOptions = field(default_factory=ScoreOptions)


class Extractor(Protocol):
    def extract(
        self, inputs: "ImageRecord"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]: ...


class _AnomalyContextProto(Protocol):
    """Protocol for context payloads passed between scorer and loss.

    We mimic a Mapping[str, Any] without inheriting from Mapping
    to satisfy Protocol constraints.
    """

    # Mapping-like methods
    def __getitem__(self, key: str) -> Any: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> KeysView[str]: ...
    def items(self) -> ItemsView[str, Any]: ...

    # Typed accessors for common fields
    @property
    def image_features(self) -> torch.Tensor | None: ...

    @property
    def patch_features_list(self) -> List[torch.Tensor] | None: ...

    @property
    def image_similarity(self) -> torch.Tensor | None: ...

    @property
    def patch_similarity_list(self) -> List[torch.Tensor] | None: ...

    @property
    def image_log_likelihoods(self) -> torch.Tensor | None: ...

    @property
    def patch_log_likelihoods_list(self) -> List[torch.Tensor] | None: ...

    @property
    def weights(self) -> torch.Tensor | None: ...


class AnomalyContext(dict, _AnomalyContextProto):
    """Mapping-based context with typed accessors for common fields."""

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)

    @property
    def image_features(self) -> torch.Tensor | None:
        return self.get("image_features")

    @property
    def patch_features_list(self) -> List[torch.Tensor] | None:
        return self.get("patch_features_list")

    @property
    def image_similarity(self) -> torch.Tensor | None:
        return self.get("image_similarity")

    @property
    def patch_similarity_list(self) -> List[torch.Tensor] | None:
        return self.get("patch_similarity_list")

    @property
    def image_log_likelihoods(self) -> torch.Tensor | None:
        return self.get("image_log_likelihoods")

    @property
    def patch_log_likelihoods_list(self) -> List[torch.Tensor] | None:
        return self.get("patch_log_likelihoods_list")

    @property
    def weights(self) -> torch.Tensor | None:
        return self.get("weights")


class AnomalyResult(TypedDict, total=False):
    context: AnomalyContext
    anomaly_score: torch.Tensor
    anomaly_map: torch.Tensor


class AnomalyScorer(Protocol):
    def compute(
        self,
        image_features: torch.Tensor,
        patch_features_list: List[torch.Tensor],
        *,
        need_context: bool,
        score_options: ScoreOptions,
    ) -> AnomalyResult: ...


class LossComputer(Protocol):
    def compute(
        self,
        result: AnomalyResult,
        label: torch.Tensor,
        gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]: ...


class PipelineOutput(TypedDict, total=False):
    image_features: torch.Tensor

    image_log_likelihoods: torch.Tensor
    patch_log_likelihoods_list: List[torch.Tensor]

    image_similarity: torch.Tensor
    patch_similarity_list: List[torch.Tensor]

    anomaly_score: torch.Tensor
    anomaly_map: torch.Tensor

    losses: Dict[str, torch.Tensor]
