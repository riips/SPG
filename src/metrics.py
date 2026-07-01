"""
metrics.py

Utility functions for computing anomaly–detection metrics such as AUROC,
average precision, PRO score, and F-beta variants.

References:
    * PRO score implementation adapted from
      https://github.com/gudovskiy/cflow-ad/blob/master/train.py
"""

from typing import Tuple, Optional, Any

import numpy as np
from numpy.typing import NDArray
from skimage import measure
from sklearn.metrics import ( # type: ignore
    auc, # type: ignore
    roc_auc_score, # type: ignore
    average_precision_score, # type: ignore
    precision_recall_curve, # type: ignore
    recall_score, # type: ignore
    precision_score, # type: ignore
    roc_curve, # type: ignore
)

def cal_pro_score(
        masks: NDArray[Any],
        amaps: NDArray[Any],
        max_step: int = 200,
        expect_fpr: float = 0.3,
    ) -> float:
    """Calculates the PRO (Per-Region Overlap) score.

    The PRO score measures the overlap between predicted anomaly maps and
    ground-truth pixel masks at a fixed false-positive-rate (FPR) threshold.
    reference: https://github.com/gudovskiy/cflow-ad/blob/master/train.py

    Args:
        masks: Binary ground-truth masks of shape ``(N, H, W)``.
        amaps: Continuous anomaly maps of shape ``(N, H, W)``.
        max_step: Number of threshold steps to sweep between
            ``amaps.min()`` and ``amaps.max()``.
        expect_fpr: The maximum FPR used to compute the area under the
            PRO-versus-FPR curve.

    Returns:
        float: Area under the PRO-FPR curve up to ``expect_fpr`` (higher is
        better).
    """
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def cal_pro_score_fast(
    masks: NDArray[Any],
    amaps: NDArray[Any],
    max_step: int = 200,
    expect_fpr: float = 0.3,
) -> float:
    """Calculates the PRO (Per-Region Overlap) score (optimized implementation).

    Same semantics as cal_pro_score but faster: precomputes labeled masks once
    and uses bincount instead of regionprops in the threshold loop.
    """
    labeled_masks = []
    for mask in masks:
        L = measure.label(mask)
        labeled_masks.append(L)

    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    inverse_masks = 1 - masks
    inv_sum = inverse_masks.sum()

    for th in np.arange(min_th, max_th, delta):
        binary_amaps = (amaps > th).astype(np.float64)
        pro_vals = []
        for i in range(len(masks)):
            L = labeled_masks[i]
            R = int(L.max())
            if R == 0:
                continue
            tp_per_region = np.bincount(
                L.ravel(), weights=binary_amaps[i].ravel(), minlength=R + 1
            )[1:]
            area_per_region = np.bincount(L.ravel(), minlength=R + 1)[1:]
            pro_vals.append((tp_per_region / area_per_region))
        if pro_vals:
            pros.append(np.concatenate(pro_vals).mean())
        else:
            pros.append(0.0)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inv_sum
        fprs.append(fpr)
        ths.append(th)

    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min() + 1e-12)
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def cal_score(
    gt: NDArray[Any],
    pr: NDArray[Any],
    metric: str,
    *,
    pro_use_fast: bool = False,
) -> Tuple[float, Optional[float]]:
    """Computes a selected metric between ground truth and predictions.

    Args:
        gt: Ground-truth array (binary for pixel- or image-level labels).
        pr: Prediction array (continuous scores in ``[0, 1]``).
        metric: Metric identifier. Supported values:

            * ``'auroc'`` – Area under ROC.
            * ``'ap'`` – Average precision.
            * ``'pro'`` – Per-Region Overlap.
            * ``'f0.5'``, ``'f1'``, ``'f2'`` – Max F-beta on PR curve.
            * ``'recall_on_f2-max'`` – Recall at threshold that maximizes F2.
            * ``'precision_on_f2-max'`` – Precision at threshold that maximizes F2.
            * ``'recall@fpr10'`` – Recall when FPR = 0.10 on ROC curve.
        pro_use_fast: When metric is ``'pro'``, use the fast implementation
            (bincount-based). If False, use the original regionprops-based implementation.

    Returns:
        tuple[float, Optional[float]]: (metric_value, threshold)
            For threshold-based metrics, returns the threshold that maximizes
            the metric. For metrics without a single operating threshold
            (e.g., AUROC/AP/PRO/recall@fpr10), the threshold is ``None``.
    """
    match metric:
        case 'auroc':
            performance = roc_auc_score(gt.ravel(), pr.ravel()) # type: ignore
            threshold = None
        
        case 'ap':
            performance = average_precision_score(gt.ravel(), pr.ravel()) # type: ignore
            threshold = None
        
        case 'pro':
            # Squeeze singleton channel dim if present (N, 1, H, W)
            if len(gt.shape) == 4:
                gt = gt.squeeze(1)
            if len(pr.shape) == 4:
                pr = pr.squeeze(1)
            if pro_use_fast:
                performance = cal_pro_score_fast(gt, pr)
            else:
                performance = cal_pro_score(gt, pr)
            threshold = None
        
        case 'f0.5' | 'f1' | 'f2':
            beta = float(metric[1:])
            precision, recall, thresholds = precision_recall_curve(gt.ravel(), pr.ravel()) # type: ignore
            # thresholds has length len(precision) - 1
            if thresholds.size == 0: # type: ignore
                performance = 0.0
                threshold = None
            else:
                a = (1 + beta**2) * precision[1:] * recall[1:]
                b = (beta**2) * precision[1:] + recall[1:]
                f_beta = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                best_idx = int(np.argmax(f_beta))
                performance = float(f_beta[best_idx])
                threshold = None if performance == 0.0 else float(thresholds[best_idx]) # type: ignore
        
        # Recall / Precision at F2-max threshold
        case 'recall_on_f2-max' | 'precision_on_f2-max':
            precision, recall, thresholds = precision_recall_curve(gt.ravel(), pr.ravel()) # type: ignore
            if thresholds.size == 0: # type: ignore
                threshold = None
                performance = 0.0
            else:
                a = (1 + 2**2) * precision[1:] * recall[1:]
                b = (2**2) * precision[1:] + recall[1:]
                f2 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                best_idx = int(np.argmax(f2))
                if float(f2[best_idx]) == 0.0:
                    threshold = None
                    performance = 0.0
                else:
                    threshold = float(thresholds[best_idx]) # type: ignore
                    rec_at = float(recall[best_idx + 1])
                    prec_at = float(precision[best_idx + 1])
                    performance = rec_at if 'recall' in metric else prec_at

        # Recall at fixed FPR = 0.10
        case 'recall@fpr10':
            fpr, tpr, threshold = roc_curve(gt.ravel(), pr.ravel())
            performance = np.interp(0.1, fpr, tpr)
            threshold = None
        
        case _:
            raise ValueError(f"Unsupported metric: {metric!r}")

    return performance, threshold # type: ignore