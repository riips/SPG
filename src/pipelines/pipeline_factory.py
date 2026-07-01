import logging
from typing import Type, assert_never

from configs import ModelCfg, PipelineMethod

from .base_pipeline import BasePipeline
from .guide_sae_pipeline import GuideSAEPipeline

log = logging.getLogger(__name__)


def get_pipeline(model_cfg: ModelCfg) -> Type[BasePipeline]:
    """
    Get pipeline class based on the method specified in the model configuration.
    Args:
        model_cfg: Model configuration namespace.
    Returns:
        Pipeline class (BasePipeline or its subclass)
    """
    method = model_cfg.method

    # convert str to PipelineMethod (Enum)
    if isinstance(method, str):
        method = PipelineMethod(method)
    log.info(f"PipelineMethod: {method}")

    # match PipelineMethod (Enum)
    match method:
        
        case PipelineMethod.GUIDE_SAE:
            log.info("GuideSAEPipeline")
            return GuideSAEPipeline
        
        case _:
            log.error(f"Invalid method: {method}")
            raise ValueError(f"Invalid method: {method}")

    # if the method is not found, raise an error
    assert_never(method, f"Invalid method: {method}")
