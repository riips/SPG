import logging
from typing import Any, Dict, List, Optional, Literal

from torch.utils.data import DataLoader

from configs import DataCfg, ModelCfg
from cache import CacheManager

from .dataset import CombinedDataset, Dataset, ImageRecord
from .preprocess import get_preprocess

log = logging.getLogger(__name__)


class DataManager:
    """Factory class for Dataset / DataLoader with optional subset filtering and dataset combination.

    Args:
        dataset_cfg: Hydra/OmegaConf config for the dataset.
        model_cfg:   Model config (needed for preprocessing).
        batch_size:  Batch size for DataLoader.
        mode:        ``"image"`` or ``"feature"``.
        cache_mgr:   Optional feature-cache manager.
        filter_kw:   Dict of conditions forwarded to ``Dataset.filter()``.
                     Example::
                         filter_kw = dict(cls_names=["bottle"], anomaly=1)
        combined_datasets: Optional list of additional dataset configs to combine.
                          Each config should have 'dataset_name', 'root', and optional 'filter_kw'.
    """

    def __init__(
        self,
        dataset_cfg: DataCfg,
        model_cfg: ModelCfg,
        batch_size: int = 1,
        *,
        mode: Literal["image", "feature"] = "image",
        cache_mgr: Optional[CacheManager] = None,
        filter_kw: Optional[Dict[str, Any]] = None,
        combined_datasets: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.dataset_name = dataset_cfg.dataset_name
        self.batch_size = batch_size
        self.mode = mode
        self.cache_mgr = cache_mgr
        self.filter_kw = filter_kw or {}
        self.combined_datasets = combined_datasets or []

        # preprocessing + collate_fn depend on backbone
        self.preprocess, self.target_transform, self.collate_fn = get_preprocess(
            model_cfg.id, model_cfg.image_size
        )
        self.dataset = self.get_dataset()

    def get_dataset(self) -> Dataset:
        """Instantiates Dataset and applies optional filtering and combination."""

        # Create main dataset
        main_dataset = Dataset(
            root=self.dataset_cfg.root,
            transform=self.preprocess,
            target_transform=self.target_transform,
            dataset_name=self.dataset_cfg.dataset_name,
            mode=self.mode,
            cache_mgr=self.cache_mgr,
        )

        # Apply filtering to main dataset
        if self.filter_kw:
            log.info("Applying dataset filter to main dataset: %s", self.filter_kw)
            main_dataset = main_dataset.filter(**self.filter_kw)
            log.info("Main dataset size after filtering: %d samples", len(main_dataset))

        # If no additional datasets to combine, return main dataset
        if not self.combined_datasets:
            return main_dataset

        # Create combined dataset
        datasets = [main_dataset]
        dataset_names = [self.dataset_cfg.dataset_name]

        for _, combined_cfg in enumerate(self.combined_datasets):
            additional_cache_mgr = CacheManager(
                {
                    "model_id": self.model_cfg.id,
                    "dataset_name": combined_cfg["dataset_name"],
                    "image_size": self.model_cfg.image_size,
                    "features_list": self.model_cfg.features_list,
                },
                self.cache_mgr.cache_dir,
            )

            # Create additional dataset
            additional_dataset = Dataset(
                root=combined_cfg["root"],
                transform=self.preprocess,
                target_transform=self.target_transform,
                dataset_name=combined_cfg["dataset_name"],
                mode=self.mode,
                cache_mgr=additional_cache_mgr,
            )

            # Apply filtering if specified
            if "filter_kw" in combined_cfg and combined_cfg["filter_kw"]:
                log.info(
                    "Applying dataset filter to %s: %s",
                    combined_cfg["dataset_name"],
                    combined_cfg["filter_kw"],
                )
                additional_dataset = additional_dataset.filter(
                    **combined_cfg["filter_kw"]
                )
                log.info(
                    "%s size after filtering: %d samples",
                    combined_cfg["dataset_name"],
                    len(additional_dataset),
                )

            datasets.append(additional_dataset)
            dataset_names.append(combined_cfg["dataset_name"])

        # Create combined dataset
        combined_dataset = CombinedDataset(datasets, dataset_names)
        log.info(
            "Combined dataset created with %d datasets, total size: %d samples",
            len(datasets),
            len(combined_dataset),
        )

        # Log dataset information
        for info in combined_dataset.get_dataset_info():
            log.info(
                "Dataset '%s': %d samples, classes: %s",
                info["name"],
                info["size"],
                info["classes"],
            )

        return combined_dataset

    def get_dataloader(
        self, shuffle: Optional[bool] = None, num_workers: int = 0
    ) -> DataLoader[ImageRecord]:
        """Returns a PyTorch DataLoader for the (sub)set."""
        return DataLoader[ImageRecord](
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if shuffle is not None else self.dataset_cfg.shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
