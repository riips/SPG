import torch
import torch.nn as nn

from configs import WeightModuleCfg

from .base_module import BaseModule


class WeightModule(BaseModule):
    """
    A module that provides adaptive weighting, supporting two modes:

    1. Dynamic weighting: Computes weights based on the input image embedding.
    2. Static weighting: Uses a learnable, fixed weight parameter.

    Set `dynamic=True` to compute weights from the input embedding (dynamic mode),
    or `dynamic=False` to use static learnable weights.
    """

    def __init__(
        self,
        cfg: WeightModuleCfg,
        model: nn.Module,
    ) -> None:
        super().__init__()  # type: ignore
        self.cfg = cfg
        self.dynamic = cfg.dynamic
        self.num_layers = cfg.num_layers

        if self.dynamic:
            # For dynamic weighting, use a linear layer to compute weights based on the input embedding.
            self.linear = nn.Linear(model.visual.output_dim, self.cfg.num_layers, bias=self.cfg.use_bias)  # type: ignore
            self._initialize_dynamic_parameters()
        else:
            # For static weighting, define a learnable weight parameter.
            self.weight = nn.Parameter(torch.empty((1, self.cfg.num_layers)))
            self._initialize_static_parameters()

    def _initialize_dynamic_parameters(self) -> None:
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.linear.bias, std=0.01)

    def _initialize_static_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.01)

    def forward(
        self, image_embedding: torch.Tensor | None = None, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass for weight computation.

        Parameters:
            image_embedding: Required for dynamic weighting mode. Ignored in static mode.
            temperature: Scaling factor applied before softmax.

        Returns:
            A tensor of weights after softmax normalization.
        """
        if self.dynamic:
            if image_embedding is None:
                raise ValueError(
                    "image_embedding must be provided when using dynamic weighting."
                )
            weight = self.linear(image_embedding / temperature).softmax(dim=-1)
        else:
            weight = (self.weight / temperature).softmax(dim=-1)
        return weight
