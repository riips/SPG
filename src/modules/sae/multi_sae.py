from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn as nn

from configs import SAECfg

from .sae_factory import create_sae
from .types import BatchOutput


class MultiSAE(nn.Module):
    """Wrapper module to manage multiple SAEs with a single API."""

    def __init__(self, sae_cfgs: List[SAECfg], input_dim: int) -> None:
        super().__init__()
        if not sae_cfgs:
            raise ValueError("sae_cfgs must contain at least one config.")

        self.sae_cfgs = list(sae_cfgs)
        self.names = self._build_names(self.sae_cfgs)
        self.sae_list = nn.ModuleList(
            [create_sae(cfg, input_dim) for cfg in self.sae_cfgs]
        )
        self._validate_input_dims()
        self.input_dim = self.sae_list[0].input_dim

    def _build_names(self, sae_cfgs: List[SAECfg]) -> List[str]:
        names: List[str] = []
        for i, cfg in enumerate(sae_cfgs):
            name = getattr(cfg, "name", None)
            if not name:
                name = f"sae{i}"
            if name in names:
                name = f"{name}_{i}"
            names.append(name)
        return names

    def _validate_input_dims(self) -> None:
        input_dims = {sae.input_dim for sae in self.sae_list}
        if len(input_dims) != 1:
            raise ValueError("All SAEs must share the same input_dim.")

    def forward(
        self, x: torch.Tensor | Mapping[str, torch.Tensor], sparsify_z: bool = True
    ) -> Dict[str, BatchOutput]:
        if isinstance(x, torch.Tensor):
            return {
                name: sae(x, sparsify_z=sparsify_z)
                for name, sae in zip(self.names, self.sae_list)
            }

        missing = [name for name in self.names if name not in x]
        if missing:
            raise ValueError(f"Missing inputs for SAE names: {missing}")
        extra = [name for name in x.keys() if name not in self.names]
        if extra:
            raise ValueError(f"Unknown SAE input names: {extra}")

        return {
            name: sae(x[name], sparsify_z=sparsify_z)
            for name, sae in zip(self.names, self.sae_list)
        }

    def enforce_unit_norm(self) -> None:
        for sae in self.sae_list:
            sae.decoder.enforce_unit_norm()

    def named_saes(self) -> Iterable[Tuple[str, nn.Module]]:
        return zip(self.names, self.sae_list)

