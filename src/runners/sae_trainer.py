"""
sae_trainer.py
"""

import logging
import os
from typing import Dict, cast

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import BaseCfg
from data import DataManager, ImageRecord
from encoders import EncoderManager
from modules.sae import SAEAnalysis, MultiSAE
from cache import CacheManager, build_feature_cache
from utils import CheckpointManager

from .base_runner import BaseRunner

log = logging.getLogger(__name__)


class SAETrainer(BaseRunner):
    def __init__(self, cfg: BaseCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

        # Load model
        self.encoder_manager = EncoderManager(cfg.model, self.device)
        self.encoder = self.encoder_manager.load_encoder()

        # Initialize cache manager (for feature reuse)
        cache_dir = os.environ.get("CACHE_DIR") or cfg.cache.dir
        self.cache_mgr = CacheManager(
            {
                "model_id": cfg.model.id,
                "dataset_name": cfg.data.dataset_name,
                "image_size": cfg.model.image_size,
                "features_list": cfg.model.features_list,
            },
            cache_dir,
        )

        # Build dataset and dataloader
        mode = cfg.data.input_type
        match mode:
            case "feature":
                if not self.cache_mgr.exists():
                    build_feature_cache(
                        self.encoder, self.device, cfg.data, cfg.model, self.cache_mgr
                    )
                cache_mgr = self.cache_mgr
            case "image":
                cache_mgr = None
            case _:
                raise ValueError(f"Invalid input type: {mode}")
        filter_kw = getattr(cfg.data, "filter_kw", None)

        self.data_mgr = DataManager(
            cfg.data,
            cfg.model,
            batch_size=cfg.train.batch_size,
            mode=mode,
            cache_mgr=cache_mgr,
            filter_kw=filter_kw,
            combined_datasets=cfg.data.combined_datasets,
        )
        self.dataloader = self.data_mgr.get_dataloader()

        # set up SAE list (zip-aligned with model.features_list)
        self.sae_cfgs = list(cfg.sae)
        if len(self.sae_cfgs) == 0:
            raise ValueError("cfg.sae must contain at least one SAE config.")
        self.feature_layers = list(cfg.model.features_list)
        if len(self.sae_cfgs) != len(self.feature_layers):
            raise ValueError(
                "len(cfg.sae) must match len(cfg.model.features_list). "
                f"got {len(self.sae_cfgs)} and {len(self.feature_layers)}"
            )

        # log.info(f"DEBUG: torch.randn(1): {torch.randn(1)}")
        self.sae = MultiSAE(self.sae_cfgs, self.encoder.output_dim)
        self.sae.to(self.device)
        self.sae_names = list(self.sae.names)
        self.sae_cfg_by_name = {
            name: cfg for name, cfg in zip(self.sae_names, self.sae_cfgs)
        }

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(), lr=cfg.train.learning_rate, betas=(0.5, 0.999)
        )

        # Scheduler setting (optional)
        self.scheduler = None
        sched_cfg = getattr(cfg.train, "scheduler", None)
        if sched_cfg:
            scheduler_cls = getattr(torch.optim.lr_scheduler, sched_cfg.type, None)
            if scheduler_cls is None:
                raise ValueError(f"Unsupported scheduler: {sched_cfg.type}")
            self.scheduler = scheduler_cls(self.optimizer, **sched_cfg.params)

        # Set up checkpoint manager
        self.checkpoint_manager = CheckpointManager(cfg.save_dir)

    def train(self) -> None:
        """
        Executes the training loop over multiple epochs.

        For each epoch:
            - Runs a single training epoch.
            - Logs the average loss at configured intervals.
            - Saves the model checkpoint at configured intervals.
            - (Optionally) Updates the learning rate scheduler.

        Returns:
            None
        """
        self.encoder.to(self.device)
        self.encoder.eval()
        self.sae.train()

        # Training loop for each epoch
        for epoch in tqdm(range(self.cfg.train.epoch), desc="Epoch"):
            loss_list = self._train_one_epoch(self.dataloader, self.device)

            # Logging
            if (epoch + 1) % self.cfg.print_freq == 0:
                _log_txt = f"epoch [{epoch + 1}/{self.cfg.train.epoch}], "
                _log_txt += ", ".join(
                    [f"{k}:{np.mean(v):.2e}" for k, v in loss_list.items()]
                )
                log.info(_log_txt)

            # Save checkpoint
            if (epoch + 1) % self.cfg.save_freq == 0:
                self.checkpoint_manager.save(epoch, sae=self.sae)

            # Update learning rate scheduler (if enabled)
            if self.scheduler:
                self.scheduler.step()

        # Compute and save dead-atom masks after training
        self.sae.eval()
        for name, sae in self.sae.named_saes():
            cfg = self.sae_cfg_by_name[name]
            analyzer = SAEAnalysis(sae, device=self.device)
            stats = analyzer.activation_stats(
                self.dataloader,
                encoder=self.encoder if self.cfg.data.input_type == "image" else None,
                features_list=(
                    self.cfg.model.features_list
                    if self.cfg.data.input_type == "image"
                    else None
                ),
                threshold=cfg.dead_atom_threshold,
                topk=None,
            )
            rule = cfg.dead_atom_rule
            if rule == "count_zero":
                dead_mask = torch.tensor(
                    stats["global"]["count_active"] == 0, dtype=torch.bool
                )
            elif rule == "usage_rate_eps":
                dead_mask = torch.tensor(
                    stats["global"]["usage_rate"] < cfg.dead_atom_eps,
                    dtype=torch.bool,
                )
            else:
                raise ValueError(f"Unsupported dead_atom_rule: {rule}")

            dead_count = int(dead_mask.sum().item())
            total_atoms = int(dead_mask.numel())
            log.info("dead_atom_mask[%s]: %d/%d", name, dead_count, total_atoms)
            torch.save(dead_mask, f"{self.cfg.save_dir}/dead_atom_mask_{name}.pt")

    def _train_one_epoch(
        self,
        dataloader: DataLoader[ImageRecord],
        device: torch.device | str,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data.
            device: Device to use (CPU or GPU).

        Returns:
            losses: Dictionary of loss lists for each loss component over the epoch.
        """

        losses: Dict[str, float] = {}
        for inputs in tqdm(dataloader):
            inputs = cast(ImageRecord, inputs)

            if isinstance(inputs.feats, torch.Tensor) and isinstance(
                inputs.mid_feats, list
            ):
                inputs.to(self.device)
                patch_features_list = inputs.mid_feats
            else:
                image = inputs.img
                with torch.no_grad():
                    _, patch_features_list = self.encoder.encode_image(
                        image=image,
                        features_list=self.cfg.model.features_list,
                    )

            if len(patch_features_list) != len(self.sae_names):
                raise ValueError(
                    "len(patch_features_list) must match len(sae). "
                    f"got {len(patch_features_list)} and {len(self.sae_names)}"
                )

            layer_inputs: Dict[str, torch.Tensor] = {}
            for name, patch_features in zip(self.sae_names, patch_features_list):
                cfg = self.sae_cfg_by_name[name]
                # If use_cls is False, drop the first (CLS) token and use only patch tokens.
                x = (
                    patch_features
                    if getattr(cfg, "use_cls", True)
                    else patch_features[:, 1:, :]
                )
                assert x.ndim == 3
                assert x.shape[0] == len(inputs.img)  # batch size
                assert x.shape[2] == self.sae.input_dim
                layer_inputs[name] = x

            outputs_dict = self.sae(layer_inputs)

            # Optimize the model parameters
            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            for name, outputs in outputs_dict.items():
                cfg = self.sae_cfg_by_name[name]
                recon_loss = outputs.recon_error.mean()
                sparsity_loss = cfg.lmbda * outputs.sparsity_penalty.mean()
                auxk_loss = outputs.aux["auxk_coef"] * outputs.aux["auxk_loss"].mean()
                loss = recon_loss + sparsity_loss + auxk_loss
                total_loss = total_loss + loss

                losses[f"recon_error/{name}"] = recon_loss.item()
                losses[f"sparsity_penalty/{name}"] = sparsity_loss.item()
                losses[f"auxk_loss/{name}"] = auxk_loss.item()
                losses[f"total_loss/{name}"] = loss.item()
            losses["total_loss"] = total_loss.item()

            total_loss.backward()  # type: ignore

            self.optimizer.step()  # type: ignore
            self.sae.enforce_unit_norm()

        return losses
