import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F  # type: ignore

from cache import CacheManager

from .image_record import ImageRecord


# --------------------------------------------------------------------------- #
#                           Class-name helpers                                #
# --------------------------------------------------------------------------- #
def generate_class_info(dataset_name: str) -> Tuple[List[str], Dict[str, int]]:
    """Returns object list and name→id mapping for a known dataset.

    Args:
        dataset_name: Identifier such as ``"mvtec"``, ``"visa"``, ``"mpdd"``,
            or any of the supported dataset tags below.

    Returns:
        Tuple[List[str], Dict[str, int]]: ``(obj_list, class_name_map_class_id)``.

    Raises:
        ValueError: If *dataset_name* is unknown.
    """
    from .meta import get_class_names
    
    obj_list = get_class_names(dataset_name)
    class_name_map_class_id: Dict[str, int] = {
        name: idx for idx, name in enumerate(obj_list)
    }
    return obj_list, class_name_map_class_id


# --------------------------------------------------------------------------- #
#                              Torch Dataset                                  #
# --------------------------------------------------------------------------- #
class Dataset(data.Dataset[ImageRecord]):
    """PyTorch `Dataset` that yields :class:`ImageRecord` objects.

    Supports two modes:

    * ``mode="image"``  - returns raw images, loads masks lazily.
    * ``mode="feature"`` - loads pre-computed feature dicts from `cache_mgr`.

    The metadata (split, anomaly label, mask path, etc.) must exist in
    ``meta.json`` located in *root*.

    Args:
        root: Dataset root directory (must contain ``meta.json``).
        transform: Callable applied to *img* before returning.
        target_transform: Callable applied to *img_mask*.
        dataset_name: High-level dataset tag (passed to `generate_class_info`).
        split: ``"train"`` or ``"test"`` section inside *meta.json*.
        mode: ``"image"`` or ``"feature"``.
        cache_mgr: Optional cache manager that implements ``path_for(idx)``.

    Note:
        The constructor **does not** perform any disk I/O besides
        reading *meta.json*.  Images and masks are read lazily in
        :pymeth:`__getitem__`.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable[[Image.Image], Union[Image.Image, torch.Tensor]]],
        target_transform: Callable[[Image.Image], torch.Tensor],
        dataset_name: str,
        *,
        split: str = "test",
        mode: str = "image",
        cache_mgr: Optional[CacheManager] = None,
    ) -> None:
        self.root = str(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.cache_mgr = cache_mgr

        # --------------------------------------------------------------- #
        # Load meta-information                                           #
        # --------------------------------------------------------------- #
        with open(Path(self.root) / "meta.json", "r") as fp:
            meta_info = json.load(fp)[split]

        self.cls_names = list(meta_info.keys())
        self.data_all: List[Dict[str, Any]] = []
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self) -> int:
        """Dataset size."""
        return self.length

    def _load_cache(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Loads cached features from disk (``mode="feature"``)."""
        assert self.cache_mgr is not None
        path = self.cache_mgr.path_for(idx)
        return torch.load(path, map_location="cpu")

    def __getitem__(self, index: int) -> ImageRecord:
        """Returns sample *index* as an :class:`ImageRecord`.

        Handles both *image* and *feature* modes transparently.

        Raises:
            FileNotFoundError: If a required image or mask file is missing.
        """
        data = self.data_all[index]

        img_path: str = data["img_path"]
        mask_path: str = data["mask_path"]
        cls_name: str = data["cls_name"]
        specie_name: str = data["specie_name"]
        anomaly: int = data["anomaly"]
        local_id: int = data["local_id"]

        # -------------------------- image ------------------------------ #
        img_file = Path(self.root) / img_path
        if not img_file.exists():
            raise FileNotFoundError(img_file)
        img = Image.open(img_file)
        img = self.transform(img) if self.transform is not None else img

        # -------------------------- mask ------------------------------- #
        mask_dir = Path(self.root) / mask_path
        if anomaly == 0 or mask_dir.is_dir():
            # no anomaly or dir placeholder → empty mask
            if isinstance(img, Image.Image):
                img_size = img.size
            else:
                # img is a tensor, infer (W, H) from tensor shape (C, H, W)
                h, w = img.shape[-2], img.shape[-1]
                img_size = (w, h)
            img_mask = Image.new("L", img_size, 0)
        else:
            mask = Image.open(mask_dir).convert("L")
            img_mask = mask.point(lambda p: 255 if p > 0 else 0)  # type: ignore

        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
        else:
            img_mask = F.to_tensor(img_mask)  # type: ignore

        # -------------------- feature mode shortcut -------------------- #
        if self.mode == "feature" and self.cache_mgr is not None:
            feats = self._load_cache(index)
            return ImageRecord(
                global_id=index,
                local_id=local_id,
                img_path=img_path,
                img=None,
                mask_path=mask_path,
                img_mask=img_mask,
                cls_id=self.class_name_map_class_id[cls_name],
                cls_name=cls_name,
                anomaly=anomaly,
                specie_name=specie_name,
                feats=feats["feats"],  # type: ignore
                mid_feats=feats["mid_feats"],  # type: ignore
            )

        # -------------------------- full record ------------------------ #
        return ImageRecord(
            global_id=index,
            local_id=local_id,
            img_path=img_path,
            img=img,
            mask_path=mask_path,
            img_mask=img_mask,
            cls_id=self.class_name_map_class_id[cls_name],
            cls_name=cls_name,
            anomaly=anomaly,
            specie_name=specie_name,
            feats=None,
            mid_feats=None,
        )

    def filter(
        self,
        *,
        cls_names: list[str] | None = None,
        anomaly: int | None = None,
        specie_names: list[str] | None = None,
        local_ids: list[int] | None = None,
    ) -> "FilteredDataset":
        """Returns a lightweight subset that preserves all Dataset behaviour.

        Args:
            cls_names: Keep only these class names.
            anomaly:   Keep only samples with this anomaly flag (0/1).
            specie_names: Keep only these defect / specie names.
            local_ids: Keep only samples with these local IDs (sequential within each class).

        Example:
            >>> bottle_ok = ds.filter(cls_names=["bottle"], anomaly=0)
            >>> bottle_specific = ds.filter(cls_names=["bottle"], local_ids=[0, 1, 2])
            >>> bottle_crack_safe = ds.filter(cls_names=["bottle"], specie_names=["crack"], local_ids=[10, 11])
            >>> for rec in DataLoader(bottle_ok, batch_size=8):
            ...     pass
        """

        def _keep(meta: Dict[str, Any]) -> bool:
            return (
                (cls_names is None or meta["cls_name"] in cls_names)
                and (anomaly is None or meta["anomaly"] == anomaly)
                and (specie_names is None or meta["specie_name"] in specie_names)
                and (local_ids is None or meta["local_id"] in local_ids)
            )

        idx = [i for i, m in enumerate(self.data_all) if _keep(m)]
        return FilteredDataset(self, idx)


class FilteredDataset(Dataset):
    """Wrapper that behaves exactly like the base Dataset but with fewer samples."""

    def __init__(self, base_ds: Dataset, idx: list[int]):
        self.base_ds = base_ds
        self.idx = idx

        # --- copy or proxy attributes needed by downstream code ----------
        self.mode = base_ds.mode

        # Build obj_list with only the classes that remain after filtering.
        self._update_class_info()

    def _update_class_info(self):
        """Update class information for the classes that remain after filtering."""
        # Collect class names that are actually present after filtering.
        actual_classes: Set[str] = set()
        for i in self.idx:
            meta = self.base_ds.data_all[i]
            actual_classes.add(meta["cls_name"])

        self.obj_list = list(actual_classes)

        # Update the class ID mapping using only the filtered classes.
        self.class_name_map_class_id: Dict[str, int] = {}
        for i, cls_name in enumerate(sorted(self.obj_list)):
            self.class_name_map_class_id[cls_name] = i

    # standard Dataset protocol
    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> ImageRecord:
        return self.base_ds[self.idx[i]]

    # fallback to original Dataset for any other attribute / method
    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_ds, name)


class CombinedDataset(Dataset):
    """Combines multiple datasets into a single dataset.

    This class allows combining multiple datasets (e.g., MVTec + GoGAN)
    or filtered subsets from different datasets.

    Args:
        datasets: List of datasets to combine
        dataset_names: Optional list of dataset names for identification
    """

    def __init__(self, datasets: list[Dataset], dataset_names: list[str] | None = None):
        self.datasets = datasets
        self.dataset_names = dataset_names or [
            f"dataset_{i}" for i in range(len(datasets))
        ]

        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for dataset in datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

        # Combine class information
        self._combine_class_info()

    def _combine_class_info(self):
        """Combines class information from all datasets."""
        # Combine obj_list (unique class names)
        all_obj_lists: List[str] = []
        for dataset in self.datasets:
            if hasattr(dataset, "obj_list"):
                all_obj_lists.extend(dataset.obj_list)
        self.obj_list = list(set(all_obj_lists))

        # Combine class_name_map_class_id
        self.class_name_map_class_id: Dict[str, int] = {}
        class_id_counter = 0
        for dataset in self.datasets:
            if hasattr(dataset, "class_name_map_class_id"):
                for cls_name, _ in dataset.class_name_map_class_id.items():
                    if cls_name not in self.class_name_map_class_id:
                        self.class_name_map_class_id[cls_name] = class_id_counter
                        class_id_counter += 1

        # Use mode from first dataset (assuming all datasets have same mode)
        self.mode = self.datasets[0].mode if self.datasets else "image"

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> ImageRecord:
        # Find which dataset contains this index
        dataset_idx = 0
        for i, cumulative_length in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cumulative_length:
                dataset_idx = i - 1
                break

        # Get the item from the appropriate dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        item = self.datasets[dataset_idx].__getitem__(local_idx)

        # Update class_id to use the combined mapping
        if hasattr(item, "cls_name") and hasattr(item, "cls_id"):
            item.cls_id = self.class_name_map_class_id.get(item.cls_name, item.cls_id)

        return item

    def get_dataset_info(self) -> List[Dict[str, Any]]:
        """Returns information about the combined datasets."""
        info: List[Dict[str, Any]] = []
        for i, (dataset, name) in enumerate(zip(self.datasets, self.dataset_names)):
            # Get the actual class information after filtering.
            actual_classes: Set[str] = set()
            if hasattr(dataset, "obj_list"):
                actual_classes = set(dataset.obj_list)

            info.append(
                {
                    "name": name,
                    "size": len(dataset),
                    "classes": list(actual_classes),
                    "start_idx": self.cumulative_lengths[i],
                    "end_idx": self.cumulative_lengths[i + 1] - 1,
                }
            )
        return info
