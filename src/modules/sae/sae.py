import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union

from .types import SparsifierKind, ReconErrorType, SparsityPenaltyType, BatchOutput


class SAEDictionary(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        atom_dim: int,
        clamp_dict_nonneg: bool = False,
        normalize_dict: bool = True,
    ):
        super().__init__()
        self._D = nn.Parameter(torch.randn(num_atoms, atom_dim) / (num_atoms**0.5))
        self.clamp_dict_nonneg = clamp_dict_nonneg
        self.normalize_dict = normalize_dict
        if self.normalize_dict:
            self.enforce_unit_norm()

    def enforce_unit_norm(self, eps: float = 1e-8):
        """Normalize dictionary atoms to unit norm in-place."""
        with torch.no_grad():
            D = self._D
            if self.clamp_dict_nonneg:
                D = F.relu(D)
            norms = D.norm(dim=-1, keepdim=True).clamp_min(eps)
            D = D / norms
            self._D.copy_(D)
    
    def _processed_weights(self) -> torch.Tensor:
        D = F.relu(self._D) if self.clamp_dict_nonneg else self._D
        if self.normalize_dict:
            D = D / (D.norm(dim=-1, keepdim=True) + 1e-8)
        return D

    @property
    def matrix(self) -> torch.Tensor:
        return self._processed_weights()

    @torch.no_grad()
    def get_atoms(self, idx) -> torch.Tensor:
        """idx: int | list | LongTensor -> returns [k, D]"""
        D = self._processed_weights()
        return D[idx]

    @torch.no_grad()
    def atom_norms(self) -> torch.Tensor:
        D = self._processed_weights()
        return D.norm(dim=-1)

# ---------- Encoder ----------
class SAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsifier_kind: str,
        sparsifier_params: Dict[str, Any],
        encoder_bias_enabled: bool = False,
    ):
        """
        Encoder for SAE: linear projection (D -> C) followed by a sparsifier.

        Args:
            input_dim (int): Input feature dim (D).
            hidden_dim (int): Code/atom dim (C).
            sparsifier_kind (str): Sparsifier name.
            sparsifier_params (Dict[str, Any]): Extra params passed to the sparsifier.

        Examples:
            SAEEncoder(D, C, sparsifier="topk", sparsifier_params={"topk": 8})
            SAEEncoder(D, C, sparsifier="relu", sparsifier_params={})
        """
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim, bias=encoder_bias_enabled)

        from .sparsifier import get_sparsifier
        self.sparsifier = get_sparsifier(sparsifier_kind, **sparsifier_params)

    def forward(self, a: torch.Tensor, sparsify_z: bool = True) -> torch.Tensor:
        _z = self.W(a)  # [B,T,C]
        if sparsify_z:
            z = self.sparsifier(_z)
        else:
            z = _z
        return z


# ---------- Decoder ----------
class SAEDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        clamp_dict_nonneg: bool = False,
        normalize_dict: bool = True,
    ):
        super().__init__()
        self.normalize_dict = normalize_dict
        self.dictionary = SAEDictionary(input_dim, output_dim, clamp_dict_nonneg, normalize_dict)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        d = self.dictionary.matrix
        rec = z @ d  # [B,T,D] @ [D,C] -> [B,T,C]
        return rec
    
    def enforce_unit_norm(self) -> None:
        self.dictionary.enforce_unit_norm()

    # --- Atom access API ---
    @torch.no_grad()
    def get_atoms(self, idx) -> torch.Tensor:
        return self.dictionary.get_atoms(idx)

    @torch.no_grad()
    def atom_norms(self) -> torch.Tensor:
        return self.dictionary.atom_norms()


# ---------- SAE core ----------
class SAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsifier_kind: str,
        sparsifier_params: Dict[str, Any] = {},
        clamp_dict_nonneg: bool = False,
        input_norm: str = "none",
        normalize_dict: bool = True,
        recon_error_type: Union[ReconErrorType, str] = ReconErrorType.MSE,
        sparsity_penalty_type: Union[SparsityPenaltyType, str] = SparsityPenaltyType.L1,
        encoder_bias_enabled: bool = False,
        tied_init: bool = False,
        auxk: int | None = None,
        auxk_coef: float = 1 / 32,
        dead_steps_threshold: int = 10_000_000,
        dead_activation_threshold: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsifier_kind = sparsifier_kind
        self.sparsifier_params = sparsifier_params
        self.clamp_dict_nonneg = clamp_dict_nonneg
        self.input_norm = input_norm
        self.normalize_dict = normalize_dict
        self.recon_error_type = ReconErrorType(recon_error_type)
        self.sparsity_penalty_type = SparsityPenaltyType(sparsity_penalty_type)
        if self.input_norm not in ("none", "l2"):
            raise ValueError(f"Unsupported input_norm: {self.input_norm}")
        
        self.encoder = SAEEncoder(input_dim, hidden_dim, self.sparsifier_kind, self.sparsifier_params, encoder_bias_enabled)
        self.decoder = SAEDecoder(hidden_dim, input_dim, self.clamp_dict_nonneg, self.normalize_dict)

        self.tied_init = tied_init
        if tied_init:
            self.encoder.W.weight.data = self.decoder.dictionary._D.data.clone()

        self.auxk = auxk
        self.auxk_coef = auxk_coef
        self.dead_steps_threshold = dead_steps_threshold
        self.dead_activation_threshold = dead_activation_threshold
        if self.auxk is not None:
            self.register_buffer(
                "stats_last_nonzero",
                torch.zeros(hidden_dim, dtype=torch.long),
                persistent=False,
            )
        else:
            self.stats_last_nonzero = None

    def encode(self, a: torch.Tensor, sparsify_z: bool = True) -> torch.Tensor:
        x = self._normalize_input(a)
        return self.encoder(x, sparsify_z=sparsify_z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor, sparsify_z: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, sparsify_z=sparsify_z)
        x_hat = self.decode(z)
        return x_hat, z

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "none":
            return x
        if self.input_norm == "l2":
            return F.normalize(x, dim=-1)
        raise ValueError(f"Unsupported input_norm: {self.input_norm}")

    # --- tokenwise reconstruction error ---
    def recon_error(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if self.recon_error_type is ReconErrorType.MSE:
            error = (x_hat - x).pow(2).mean(dim=-1)  # [B,T]
        elif self.recon_error_type is ReconErrorType.COS:
            x_n = F.normalize(x, dim=-1)
            xh_n = F.normalize(x_hat, dim=-1)
            cos = (x_n * xh_n).mean(dim=-1)  # [B,T]
            error = 1 - cos 
        else:
            raise ValueError(f"Unsupported reconstruction error type: {self.recon_error_type}")
        return error

    # --- tokenwise sparsity penalty ---
    def sparsity_penalty(self, z: torch.Tensor) -> torch.Tensor:
        if self.sparsity_penalty_type is SparsityPenaltyType.L1:
            penalty = z.abs().mean(dim=-1)  # [B,T]
        elif self.sparsity_penalty_type is SparsityPenaltyType.L0_PROXY:
            penalty = (z > 0).float().mean(dim=-1)  # [B,T]
        else:
            raise ValueError(f"Unsupported sparsity penalty type: {self.sparsity_penalty_type}")
        return penalty

    def forward(self, x: torch.Tensor, sparsify_z: bool = True) -> BatchOutput:
        x_in = self._normalize_input(x)
        pre = self.encoder.W(x_in)
        if sparsify_z:
            z = self.encoder.sparsifier(pre)
        else:
            z = pre
        x_hat = self.decode(z)
        
        recon_error = self.recon_error(x_in, x_hat)
        sparsity_penalty = self.sparsity_penalty(z)

        aux: Dict[str, Any] = {}
        if self.auxk is not None:
            kaux = min(self.auxk, pre.shape[-1])
            if kaux > 0 and self.stats_last_nonzero is not None:
                with torch.no_grad():
                    active = (z > self.dead_activation_threshold).any(dim=(0, 1))
                    active_long = active.to(self.stats_last_nonzero.dtype)
                    self.stats_last_nonzero *= (1 - active_long)
                    self.stats_last_nonzero += 1
                    step_tokens = int(pre.shape[0] * pre.shape[1])
                    dead_steps = max(1, int(self.dead_steps_threshold // step_tokens))
                    dead_mask = self.stats_last_nonzero > dead_steps

                if dead_mask.any():
                    pre_dead = pre * dead_mask
                    auxk_vals, auxk_idx = torch.topk(pre_dead, kaux, dim=-1)
                    auxk_vals = auxk_vals.clamp_min(0)
                    z_aux = torch.zeros_like(pre)
                    z_aux.scatter_(-1, auxk_idx, auxk_vals)

                    e = x_in - x_hat.detach()
                    e_hat = self.decode(z_aux)
                    auxk_loss = (e - e_hat).pow(2).mean(dim=-1)
                    auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)

                    aux = {
                        "auxk_loss": auxk_loss,
                        "auxk_dead_mask": dead_mask,
                        "auxk_indices": auxk_idx,
                        "auxk_values": auxk_vals,
                        "auxk_coef": self.auxk_coef,
                    }
                else:
                    aux = {
                        "auxk_loss": torch.zeros_like(recon_error),
                        "auxk_dead_mask": dead_mask,
                        "auxk_coef": self.auxk_coef,
                    }
        else:
            aux = {
                "auxk_loss": torch.zeros_like(recon_error),
                "auxk_dead_mask": torch.zeros(pre.shape[-1], dtype=torch.bool, device=pre.device),
                "auxk_coef": self.auxk_coef,
            }

        return BatchOutput(
            x=x_in,
            z=z,
            x_hat=x_hat,
            recon_error=recon_error,
            sparsity_penalty=sparsity_penalty,
            sparsify_z=sparsify_z,
            aux=aux,
        )

    # --- Delegate dictionary atom API ---
    @torch.no_grad()
    def get_atoms(self, idx):
        return self.decoder.get_atoms(idx)

    @torch.no_grad()
    def atom_norms(self):
        return self.decoder.atom_norms()
