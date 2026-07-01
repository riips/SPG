# modules/sae/sae_factory.py

from .sae import SAE
from configs import SAECfg

def create_sae(cfg: SAECfg, input_dim: int) -> SAE:
    """
    Create a SAE model from a configuration.

    Args:
        cfg: SAECfg configuration.
        input_dim: Input feature dimension.

    Returns:
        SAE model.
    """
    return SAE(
        input_dim=input_dim, 
        hidden_dim=cfg.hidden_dim,
        sparsifier_kind=cfg.sparsifier_kind,
        sparsifier_params=cfg.sparsifier_params,
        clamp_dict_nonneg=cfg.clamp_dict_nonneg,
        input_norm=cfg.input_norm,
        recon_error_type=cfg.recon_error_type,
        sparsity_penalty_type=cfg.sparsity_penalty_type,
        encoder_bias_enabled=cfg.encoder_bias_enabled,
        tied_init=cfg.tied_init,
        auxk=cfg.auxk,
        auxk_coef=cfg.auxk_coef,
        dead_steps_threshold=cfg.dead_steps_threshold,
        dead_activation_threshold=cfg.dead_activation_threshold,
    )