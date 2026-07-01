import logging
import random

import numpy as np
import torch

from configs import BaseCfg

log = logging.getLogger(__name__)


class BaseRunner:
    def __init__(self, cfg: BaseCfg) -> None:
        """Initialize the base runner."""
        self._setup_seed(cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_seed(self, seed: int) -> None:
        """Set up seed for reproducibility."""
        torch.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
