from abc import ABC, abstractmethod
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import ModelCfg
from data import ImageRecord
from encoders.base_encoder import BaseEncoder

from .core.contracts import (
    AnomalyContext,
    AnomalyScorer,
    Extractor,
    ForwardOptions,
    LossComputer,
    PipelineOutput,
    ReturnFields,
    ScoreOptions,
)

log = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Backbone-agnostic pipeline skeleton.

    Args:
        model_cfg: Model configuration namespace.
        encoder: Frozen backbone network (e.g. CLIP, ViT).
        device: Torch device on which to run computations.
    """

    def __init__(
        self, model_cfg: ModelCfg, encoder: BaseEncoder, device: torch.device | str
    ) -> None:
        """Initialize the base pipeline."""
        self.cfg = model_cfg
        self.encoder = encoder
        self.device = torch.device(device)
        self._modules = nn.ModuleDict()
        self.extractor, self.anomaly_scorer, self.loss_computer = self.build_components(
            model_cfg, encoder
        )

    @abstractmethod
    def build_components(
        self, model_cfg: ModelCfg, encoder: BaseEncoder
    ) -> Tuple[Extractor, AnomalyScorer, LossComputer]:
        raise NotImplementedError

    def forward(
        self,
        inputs: ImageRecord,
        options: ForwardOptions | None = None,
        *,
        get_features: bool = False,
        get_similarity: bool = False,
        get_score: bool = False,
        get_loss: bool = True,
        aggregate_anomaly_map: bool = True,
    ) -> PipelineOutput:

        # default values for backward compatibility
        if options is None:
            returns = ReturnFields.NONE
            if get_features:
                returns |= ReturnFields.FEATURES
            if get_similarity:
                returns |= ReturnFields.SIMILARITY
            if get_score:
                returns |= ReturnFields.SCORE
            if get_loss:
                returns |= ReturnFields.LOSS
            score_options = ScoreOptions(aggregate_map=aggregate_anomaly_map)
            options = ForwardOptions(returns=returns, score_options=score_options)

        outputs: PipelineOutput = {}

        # (1) extract features
        image_features, patch_features_list = self.extractor.extract(inputs)

        # (2) compute anomaly score and anomaly map
        need_context = bool(
            options.returns
            & (ReturnFields.FEATURES | ReturnFields.SIMILARITY | ReturnFields.LOSS | ReturnFields.SCORE)
        )
        result = self.anomaly_scorer.compute(
            image_features,
            patch_features_list,
            need_context=need_context,
            score_options=options.score_options,
        )

        # (3) collect optional outputs
        if need_context:
            ctx: AnomalyContext = result.get("context", AnomalyContext())
            outputs["context"] = ctx

            if options.returns & ReturnFields.FEATURES:
                assert (
                    "image_features" in ctx
                ), "image_features is required when need_context is True"
                outputs["image_features"] = ctx["image_features"].detach().cpu()

            if options.returns & ReturnFields.SIMILARITY:
                assert (
                    "image_similarity" in ctx
                ), "image_similarity is required when need_context is True"
                assert (
                    "patch_similarity_list" in ctx
                ), "patch_similarity_list is required when need_context is True"
                outputs["image_similarity"] = ctx["image_similarity"].detach().cpu()
                outputs["patch_similarity_list"] = [
                    s.detach().cpu() for s in ctx["patch_similarity_list"]
                ]
            
            if options.returns & ReturnFields.SCORE:
                assert "anomaly_score" in result, "anomaly_score is required when need_context is True"
                assert "anomaly_map" in result, "anomaly_map is required when need_context is True"
                outputs["anomaly_score"] = result["anomaly_score"].detach().cpu()
                outputs["anomaly_map"] = result["anomaly_map"].detach().cpu()

        # (4) compute loss
        if options.returns & ReturnFields.LOSS:
            if isinstance(inputs.anomaly, torch.Tensor):
                label = inputs.anomaly.clone().detach()
                label = label.to(dtype=torch.long, device=self.device)
            else:
                label = torch.tensor(
                    inputs.anomaly, dtype=torch.long, device=self.device
                )
            gt = torch.where(inputs.img_mask.squeeze().to(self.device) > 0.5, 1, 0)

            loss_dict = self.loss_computer.compute(result, label, gt)
            outputs["losses"] = loss_dict

        return outputs

    def get_modules(self) -> nn.ModuleDict:
        return self._modules

    def get_params(self) -> List[torch.nn.Parameter]:
        return list(self._modules.parameters())

    def post_step(self) -> None:
        pass

    def finalize(self, dataloader: DataLoader[ImageRecord]) -> None:
        pass