"""image_record.py

Data-container object that bundles all metadata and tensors related to a single
image sample.  The record is kept lightweight by relying on the `dataclass`
decorator, while still supporting `.to(device)` for easy Tensor relocation.
"""

from dataclasses import dataclass, fields
from typing import Optional, Union, List
import logging

import torch
from PIL import Image


log = logging.getLogger(__name__)


class VisualMixin:
    """Adds lightweight visualization helpers to :class:`ImageRecord`."""

    def show(
        self,
        with_mask: bool = True,
        cmap_mask: str = "gray",
        figsize: tuple[int, int] = (6, 3),
        title: str | None = None,
    ) -> None:
        """Displays the image (left) and, optionally, its mask (right).

        Args:
            with_mask: If ``True`` and a mask is present, show it in a second
                panel.  If no mask is available, the right panel is omitted.
            cmap_mask: Colormap used for the mask.
            figsize:  Matplotlib figure size ``(w, h)`` in inches.
            title: Optional figure title.

        Notes:
            *The import of ``matplotlib`` and ``torchvision`` is deferred* so
            that headless environments can still import this module.
        """
        import matplotlib.pyplot as plt
        from torchvision.transforms.functional import to_pil_image  # type: ignore

        # ----------------------------- prepare --------------------------- #
        def _to_pil(t: torch.Tensor) -> Image.Image:
            return to_pil_image(t.cpu()) if isinstance(t, torch.Tensor) else t  # type: ignore

        img_pil = _to_pil(self.img)  # type: ignore
        if self.img is None:
            img_pil = Image.open(self.root / self.img_path)
        mask_pil = _to_pil(self.img_mask) if self.img_mask is not None else None  # type: ignore

        show_mask = with_mask and mask_pil is not None

        # --------------------------- draw figure ------------------------- #
        n_cols = 2 if show_mask else 1
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)  # type: ignore
        ax_img = axes[0, 0]
        ax_img.imshow(img_pil)
        ax_img.set_title("Image")
        ax_img.axis("off")

        if show_mask:
            ax_mask = axes[0, 1]
            ax_mask.imshow(mask_pil, cmap=cmap_mask)
            ax_mask.set_title("Mask")
            ax_mask.axis("off")

        if title:
            fig.suptitle(title, fontsize=12)  # type: ignore

        plt.tight_layout()
        plt.show()  # type: ignore


@dataclass
class ImageRecord(VisualMixin):
    """Holds raw paths, class labels, and backbone features for one sample.

    Attributes:
        global_id: Index over the entire dataset.
        local_id:  Index **within** the class.
        img_path: Path to the original image file.
        img: Loaded image tensor ``(C, H, W)`` (optional; may be loaded lazily).
        mask_path: Path to the ground-truth mask file, if available.
        img_mask: Ground-truth mask tensor ``(H, W)`` or ``(1, H, W)``.
        cls_id: Integer or 1-element tensor with class index.
        cls_name: Human-readable class name (e.g. ``"screw"``).
        anomaly: 0/1 integer or tensor (or one-hot) indicating anomaly.
        specie_name: Sub-class or defect type (e.g. ``"zipper"``).
        feats: Global feature vector from the backbone ``(D,)``.
        mid_feats: List of mid-layer feature maps (e.g. patch tokens).
    """

    # ----------------------------- basic ids ---------------------------- #
    global_id: int
    local_id: int

    # --------------------------- file handles --------------------------- #
    img_path: str
    img: Union[torch.Tensor, Image.Image, None]

    mask_path: str
    img_mask: torch.Tensor

    # --------------------------- class labels --------------------------- #
    cls_id: Union[int, torch.Tensor]
    cls_name: str
    anomaly: Union[int, torch.Tensor]
    specie_name: str

    # ---------------------- backbone feature outputs -------------------- #
    feats: Optional[torch.Tensor]  # Final feature vector
    mid_feats: Optional[List[torch.Tensor]]  # Intermediate-layer feature maps

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        indent = "  "
        body = ",\n".join(
            f"{indent}{f.name} = {getattr(self, f.name)!r}" for f in fields(self)
        )
        return f"{cls_name}(\n{body}\n)"

    def to(self, device: torch.device | str) -> "ImageRecord":
        """Moves all stored tensors to *device* (in-place).

        Args:
            device: Target device, e.g. ``"cpu"``, ``"cuda:0"``.

        Returns:
            ImageRecord: Self, to enable method chaining.
        """

        if self.feats is not None:
            self.feats = self.feats.to(device)

        if self.mid_feats is not None:
            self.mid_feats = [f.to(device) for f in self.mid_feats]

        return self
