"""
trainer.py
"""

import logging
import os
from typing import Dict, List

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import BaseCfg
from data import DataManager, ImageRecord
from encoders import EncoderManager
from pipelines.pipeline_factory import get_pipeline
from pipelines.core.contracts import (
    PipelineOutput,
    ForwardOptions,
    ReturnFields,
    ScoreOptions,
)
from cache import CacheManager, build_feature_cache
from utils import CheckpointManager

from .base_runner import BaseRunner

log = logging.getLogger(__name__)


class Trainer(BaseRunner):
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

        # Build processing pipeline
        self.pipeline = get_pipeline(cfg.model)(cfg.model, self.encoder, self.device)
        self.modules = self.pipeline.get_modules()
        self.modules.to(self.device)
        self.params = self.pipeline.get_params()

        self.ema_model: AveragedModel | None = None
        ema_cfg = getattr(cfg.train, "ema", None)
        if ema_cfg and ema_cfg.enabled:
            self.ema_model = AveragedModel(
                self.modules,
                multi_avg_fn=get_ema_multi_avg_fn(ema_cfg.decay),
                use_buffers=ema_cfg.use_buffers,
            )

        self.optimizer = torch.optim.Adam(
            self.params, lr=cfg.train.learning_rate, betas=(0.5, 0.999)
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

        self.global_step = 0

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
        self.modules.train()

        # Training loop over epochs
        self.global_step = 0
        for epoch in tqdm(range(self.cfg.train.epoch), desc="Epoch"):
            loss_list = self._train_one_epoch(self.dataloader, self.device)

            # Logging
            if (epoch + 1) % self.cfg.print_freq == 0:
                _log_txt = f"epoch [{epoch + 1}/{self.cfg.train.epoch}], "
                _log_txt += ", ".join(
                    [f"{k}:{np.mean(v):.4f}" for k, v in loss_list.items()]
                )
                log.info(_log_txt)

            # Save checkpoint
            if (epoch + 1) % self.cfg.save_freq == 0:
                if self.ema_model is None:
                    self.checkpoint_manager.save(epoch, modules_dict=self.modules)
                else:
                    self.checkpoint_manager.save(
                        epoch,
                        modules_dict=self.modules,
                        ema_modules_dict=self.ema_model.module,
                    )            

            # Update learning rate scheduler (if enabled)
            if self.scheduler:
                self.scheduler.step()

    def _train_one_epoch(
        self,
        dataloader: DataLoader[ImageRecord],
        device: torch.device | str,
    ) -> Dict[str, List[float]]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data.
            device: Device to use (CPU or GPU).

        Returns:
            losses: Dictionary of loss lists for each loss component over the epoch.
        """

        losses: Dict[str, List[float]] = {}
        opts = ForwardOptions(
            returns=ReturnFields.LOSS, score_options=ScoreOptions(aggregate_map=True)
        )
        for inputs in tqdm(dataloader):
            outputs: PipelineOutput = self.pipeline.forward(inputs, options=opts)

            loss_dict = outputs.get("losses", {})

            # Track all loss components for logging
            for k, v in loss_dict.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v.item())

            # Optimize the model parameters
            self.optimizer.zero_grad()
            if len(loss_dict) == 0:
                continue
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.backward()  # type: ignore

            self.optimizer.step()  # type: ignore
            if self.ema_model is not None:
                ema_cfg = self.cfg.train.ema
                warmup_steps = getattr(ema_cfg, "warmup_steps", 0) if ema_cfg else 0
                if self.global_step >= warmup_steps:
                    self.ema_model.update_parameters(self.modules)
            self.pipeline.post_step()
            self.global_step += 1

        return losses
