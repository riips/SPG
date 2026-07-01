"""
Utility script for visualizing and analyzing SAE atoms.

This script provides the following subcommands:
- ``individual``: render overlay tiles for a specified list of atom IDs.
- ``batch_siglip``: analyze and render overlays for all active atoms of a SigLIP SAE.
- ``batch_dino``: analyze and render overlays for all active atoms of a DINO SAE.

If the script is called without an explicit subcommand but with ``--atoms`` (legacy usage),
the arguments are interpreted as the ``individual`` subcommand.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from modules.sae import SAE, SAEAnalysis
from utils import CheckpointManager

from analysis_sae.common import load_sae_compat, make_log

ROOTS = {
    "mvtec": "../../datasets/mvtec_anomaly_detection",
    "visa": "../../datasets/visa",
}
DATASET_ROOT_DIRS = {
    "mvtec": "mvtec_anomaly_detection",
    "visa": "visa",
}

DEFAULT_DINO_CKPT = "outputs/2026-02-03/20-36-06_none_facebook_dinov3-vitl16/checkpoints/epoch_50.pth"

# default setting
DEFAULT_SIGLIP_HIDDEN_DIM = 4096
DEFAULT_SIGLIP_TOPK = 32
DEFAULT_SIGLIP_CKPT = "outputs/2026-02-03/18-20-23_none_google_siglip-so400m-patch14-384/checkpoints/epoch_50.pth"

DEFAULT_DINO_HIDDEN_DIM = 4096
DEFAULT_DINO_TOPK = 32

INDIVIDUAL_CKPT_MANAGER_SAVE_DIR = "outputs/2025-11-19/12-04-54_none_google_siglip-so400m-patch14-384/"
BATCH_CKPT_MANAGER_SAVE_DIR = "outputs/2025-11-17/12-04-54_none_google_siglip-so400m-patch14-384/"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "features")
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cuda"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    id_wo_slash: str
    encoder: str
    input_dim: int
    image_size: int
    features_list: list[int]
    cache_features_list: list[int]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str


@dataclass(frozen=True)
class OverlaySpec:
    cols_per_atom: int = 5
    topn: int = 50
    per_class_topk: int = 10
    figsize: tuple[int, int] = (10, 20)
    show: bool = False


MODEL_SPECS = {
    "dino": ModelSpec(
        id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        id_wo_slash="facebook_dinov3-vitl16",
        encoder="DINO",
        input_dim=1024,
        image_size=448,
        features_list=[24],
        cache_features_list=[24],
    ),
    "siglip": ModelSpec(
        id="google/siglip-so400m-patch14-384",
        id_wo_slash="google_siglip-so400m-patch14-384",
        encoder="SigLIP",
        input_dim=1152,
        image_size=384,
        features_list=[6, 13, 20, 27],
        cache_features_list=[24],
    ),
}

DEFAULT_OVERLAY_SPEC = OverlaySpec()


def _ls_png(p: Path) -> set[str]:
    return {f.name for f in p.glob("*.png")}


def _parse_int_list_csv(s: str) -> list[int]:
    return [int(a) for a in s.split(",") if a.strip()]


def _dataset_spec(dataset_name: str, root: str | None = None) -> DatasetSpec:
    if root is not None:
        return DatasetSpec(name=dataset_name, root=root)

    dataset_root = os.environ.get("DATASET_ROOT")
    if dataset_root and dataset_name in DATASET_ROOT_DIRS:
        return DatasetSpec(name=dataset_name, root=str(Path(dataset_root) / DATASET_ROOT_DIRS[dataset_name]))

    if dataset_name not in ROOTS:
        available = sorted(set(ROOTS) | set(DATASET_ROOT_DIRS))
        raise KeyError(f"Unsupported dataset: {dataset_name}. Available: {available}")
    return DatasetSpec(name=dataset_name, root=ROOTS[dataset_name])


def _cache_dir(cache_dir: str | None = None) -> str:
    return cache_dir or os.environ.get("CACHE_DIR") or DEFAULT_CACHE_DIR


def _checkpoint_id(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if len(p.parents) >= 3:
        return f"{p.parent.parent.parent.name}_{p.parent.parent.name}"
    return p.stem


def _individual_out_dir(
    ckpt_path: str,
    dataset_name: str,
    topk: int,
    hidden_dim: int,
    output_dir: str | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    base = Path(f"outputs/sae_results/topk{topk}_dim{hidden_dim}_dinov3_individual")
    return base / _checkpoint_id(ckpt_path) / dataset_name


def _batch_out_dir(model_key: str, topk: int, hidden_dim: int, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    suffix = "_dinov3" if model_key == "dino" else ""
    return Path(f"outputs/sae_results/topk{topk}_dim{hidden_dim}{suffix}")


def _build_sae(
    *,
    model: ModelSpec,
    hidden_dim: int,
    topk: int,
    input_norm: str | None = None,
) -> SAE:
    kwargs = {
        "input_dim": model.input_dim,
        "hidden_dim": hidden_dim,
        "sparsifier_kind": "topk",
        "sparsifier_params": {"topk": topk},
        "recon_error_type": "mse",
        "sparsity_penalty_type": "l1",
    }
    if input_norm is not None:
        kwargs["input_norm"] = input_norm
    return SAE(**kwargs)


def _build_data_manager(
    *,
    dataset: DatasetSpec,
    model: ModelSpec,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: str | None = None,
):
    from cache import CacheManager
    from configs import DataCfg, ModelCfg
    from data import DataManager

    cache_mgr = CacheManager(
        key={
            "model_id": model.id,
            "dataset_name": dataset.name,
            "image_size": model.image_size,
            "features_list": model.cache_features_list,
        },
        cache_dir=_cache_dir(cache_dir),
    )

    return DataManager(
        dataset_cfg=DataCfg(
            dataset_name=dataset.name,
            input_type="feature",
            path="",
            root=dataset.root,
            filter_kw=None,
            combined_datasets=None,
            shuffle=True,
        ),
        model_cfg=ModelCfg(
            id=model.id,
            id_wo_slash=model.id_wo_slash,
            encoder=model.encoder,
            image_size=model.image_size,
            features_list=model.features_list,
            method=None,
            method_config=None,
        ),
        batch_size=batch_size,
        mode="feature",
        cache_mgr=cache_mgr,
    )


def _stats_csv_path(out_dir: Path, log: Callable[[str], None]) -> Path:
    stats_csv = out_dir / "per_atom_stats.csv"
    if not stats_csv.exists():
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "atom_id",
                    "num_candidates",
                    "num_drawn",
                    "num_positive",
                    "distinct_classes_positive",
                    "class_counts_positive_json",
                ]
            )
        log(f"[INFO] Created stats csv: {stats_csv}")
    return stats_csv


def _append_per_atom_stats(stats_csv: Path, per_atom: dict) -> int:
    n_rows = 0
    if not per_atom:
        return n_rows

    with open(stats_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for a, st in per_atom.items():
            class_counts = st.get("class_counts_positive", {}) or {}
            writer.writerow(
                [
                    int(a),
                    int(st.get("num_candidates", 0)),
                    int(st.get("num_drawn", 0)),
                    int(st.get("num_positive", 0)),
                    int(st.get("distinct_classes_positive", 0)),
                    json.dumps(class_counts, ensure_ascii=False),
                ]
            )
            n_rows += 1
    return n_rows


def _log_activation_stats(log: Callable[[str], None], stats: dict) -> None:
    usage = stats["global"].get("usage_rate", None)
    cnt_active = stats["global"].get("count_active", None)
    cnt_tokens = stats["global"].get("count_tokens", None)
    if usage is not None:
        log(f"[STATS] usage_rate: mean={np.mean(usage):.6f}, max={np.max(usage):.6f}, min={np.min(usage):.6f}")
    if cnt_active is not None:
        log(f"[STATS] count_active: total={np.sum(cnt_active):.0f}, max={np.max(cnt_active):.0f}, nonzero={(np.asarray(cnt_active) > 0).sum()}")
    if cnt_tokens is not None:
        log(f"[STATS] count_tokens (global): {cnt_tokens}")


def _run_active_atom_overlay_chunks(
    *,
    sae_analysis: SAEAnalysis,
    dataloader,
    active_atom_ids: np.ndarray,
    chunk_size: int,
    out_dir: Path,
    log: Callable[[str], None],
    overlay: OverlaySpec = DEFAULT_OVERLAY_SPEC,
) -> None:
    before_all = _ls_png(out_dir)
    stats_csv = _stats_csv_path(out_dir, log)

    for i in range(0, len(active_atom_ids), chunk_size):
        atom_ids = active_atom_ids[i : i + chunk_size]
        log(
            f"[RUN] chunk {i // chunk_size}: atoms[{i}:{i + len(atom_ids)}] -> {list(atom_ids[:5])}{'...' if len(atom_ids) > 5 else ''} (n={len(atom_ids)})"
        )
        tc0 = time.perf_counter()
        before = _ls_png(out_dir)

        overlay_stats = sae_analysis.show_top_overlays_for_atoms_streaming(
            dataloader,
            atom_ids=atom_ids,
            cols_per_atom=overlay.cols_per_atom,
            topn=overlay.topn,
            figsize=overlay.figsize,
            save_dir=str(out_dir),
            show=overlay.show,
        )

        n_rows = _append_per_atom_stats(stats_csv, overlay_stats.get("per_atom", {}))
        after = _ls_png(out_dir)
        new_files = sorted(list(after - before))
        tc1 = time.perf_counter()
        log(f"[DONE] chunk {i // chunk_size}: created {len(new_files)} files in {tc1 - tc0:.2f}s")
        if new_files:
            log(f"[FILES] sample: {new_files[:3]}")
        log(f"[STATS] chunk {i // chunk_size}: wrote {n_rows} rows to {stats_csv.name}")

    after_all = _ls_png(out_dir)
    log(f"[SUMMARY] total new files: {len(after_all - before_all)} -> out_dir={out_dir}")


def _inject_compat_subcommand(argv: list[str]) -> list[str]:
    """
    Backward-compatible handling of legacy CLI calls.

    If the script is invoked without an explicit subcommand (for example via
    ``python -m analysis_sae.sae_atom_overlays --atoms ...``), the arguments
    are interpreted as the ``individual`` subcommand.
    """
    modes = {"individual", "batch_siglip", "batch_dino"}
    if len(argv) >= 1:
        first = argv[0]
        if first not in modes and first.startswith("-"):
            return ["individual", *argv]
    return argv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize and analyze SAE atoms.")
    sub = p.add_subparsers(dest="mode", required=True)

    # --------------------
    # individual
    # --------------------
    ind = sub.add_parser("individual", description="Render overlays for specific atom IDs.")
    ind.add_argument(
        "--atoms",
        nargs="*",
        type=int,
        default=[0, 1, 42],
        help="List of atom IDs to visualize (space separated).",
    )
    ind.add_argument("--ckpt", type=str, default=DEFAULT_DINO_CKPT, help="Path to SAE checkpoint (.pth).")
    ind.add_argument("--hidden-dim", type=int, default=DEFAULT_DINO_HIDDEN_DIM, dest="hidden_dim", help="SAE hidden_dim.")
    ind.add_argument("--topk", type=int, default=DEFAULT_DINO_TOPK, help="SAE topk.")
    ind.add_argument("--dataset", type=str, default="mvtec", help="Dataset name (passed to DataCfg / CacheManager).")
    ind.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", type=str, default=None, help="Dataset root path.")
    ind.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None, help="Feature cache directory.")
    ind.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None, help="Output directory.")
    ind.add_argument(
        "--input-norm",
        type=str,
        choices=("none", "l2"),
        default="l2",
        dest="input_norm",
        help="SAE input normalization (none / l2).",
    )

    # --------------------
    # batch_siglip
    # --------------------
    bs = sub.add_parser("batch_siglip", description="Analyze all active atoms for a SigLIP SAE.")
    bs.add_argument("--ckpt", type=str, default=DEFAULT_SIGLIP_CKPT, help="Path to SAE checkpoint (.pth).")
    bs.add_argument("--hidden-dim", type=int, default=DEFAULT_SIGLIP_HIDDEN_DIM, dest="hidden_dim", help="SAE hidden_dim.")
    bs.add_argument("--topk", type=int, default=DEFAULT_SIGLIP_TOPK, help="SAE topk.")
    bs.add_argument("--dataset", type=str, default="mvtec", help="Dataset name (passed to DataCfg / CacheManager).")
    bs.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", type=str, default=None, help="Dataset root path.")
    bs.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None, help="Feature cache directory.")
    bs.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None, help="Output directory.")
    bs.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Chunk size for atom IDs.",
    )

    # --------------------
    # batch_dino
    # --------------------
    bd = sub.add_parser("batch_dino", description="Analyze all active atoms for a DINO SAE.")
    bd.add_argument("--ckpt", type=str, default=DEFAULT_DINO_CKPT, help="Path to SAE checkpoint (.pth).")
    bd.add_argument("--hidden-dim", type=int, default=DEFAULT_DINO_HIDDEN_DIM, dest="hidden_dim", help="SAE hidden_dim.")
    bd.add_argument("--topk", type=int, default=DEFAULT_DINO_TOPK, help="SAE topk.")
    bd.add_argument("--dataset", type=str, default="mvtec", help="Dataset name (passed to DataCfg / CacheManager).")
    bd.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", type=str, default=None, help="Dataset root path.")
    bd.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None, help="Feature cache directory.")
    bd.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None, help="Output directory.")
    bd.add_argument(
        "--analysis-features-list",
        type=str,
        default=None,
        help="Comma-separated features_list for activation_stats. Defaults to ModelCfg.features_list (= [24]).",
    )
    bd.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Chunk size for atom IDs.",
    )

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    argv = _inject_compat_subcommand(list(argv))
    return build_parser().parse_args(argv)


def run_individual(args: argparse.Namespace) -> None:
    hidden_dim: int = args.hidden_dim
    topk: int = args.topk
    ckpt_path: str = args.ckpt
    dataset_name: str = args.dataset
    input_norm: str = args.input_norm
    atom_ids: list[int] = [int(a) for a in args.atoms]
    model = MODEL_SPECS["dino"]
    dataset_spec = _dataset_spec(dataset_name, args.dataset_root)

    sae = _build_sae(model=model, hidden_dim=hidden_dim, topk=topk, input_norm=input_norm)

    # load_sae_compat only uses `ckpt_mgr.device`, so a dummy save_dir is enough.
    ckpt_mgr = CheckpointManager(save_dir=INDIVIDUAL_CKPT_MANAGER_SAVE_DIR)
    load_sae_compat(ckpt_mgr, ckpt_path, sae)
    sae_analysis = SAEAnalysis(sae, DEFAULT_DEVICE)

    # Output directory for individual tiles, separated from bulk outputs and
    # organized by checkpoint -> dataset.
    out_dir = _individual_out_dir(ckpt_path, dataset_name, topk, hidden_dim, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_log(out_dir / "run.log")

    log(f"[INFO] SAE loaded: hidden_dim={hidden_dim}, sparsifier=topk({topk}), input_norm={input_norm}, device=cuda")
    log(f"[INFO] Checkpoint: {ckpt_path}")
    log(f"[INFO] Dataset: name={dataset_spec.name}, root={dataset_spec.root}")

    data_mgr = _build_data_manager(dataset=dataset_spec, model=model, cache_dir=args.cache_dir)

    dataset = data_mgr.get_dataset()
    dataloader = data_mgr.get_dataloader(shuffle=False)
    log(f"[INFO] DataLoader prepared: size={len(dataset)}, batch_size={data_mgr.batch_size}")
    log(f"[INFO] Target atom IDs: {atom_ids} (n={len(atom_ids)})")

    t0 = time.perf_counter()
    overlay_stats = sae_analysis.show_top_overlays_for_atoms_streaming(
        dataloader,
        root=dataset_spec.root,
        atom_ids=atom_ids,
        cols_per_atom=DEFAULT_OVERLAY_SPEC.cols_per_atom,
        topn=DEFAULT_OVERLAY_SPEC.topn,
        per_class_topk=DEFAULT_OVERLAY_SPEC.per_class_topk,
        figsize=DEFAULT_OVERLAY_SPEC.figsize,
        save_dir=str(out_dir),
        show=DEFAULT_OVERLAY_SPEC.show,
        save_individual_tiles=True,
        save_classwise_tiles=True,
    )
    t1 = time.perf_counter()

    per_atom = overlay_stats.get("per_atom", {})
    total_drawn = sum(st.get("num_drawn", 0) for st in per_atom.values())
    log(f"[DONE] overlays done in {t1 - t0:.2f}s")
    log(f"[SUMMARY] atoms_processed={len(per_atom)}, total_tiles_saved={total_drawn}, out_dir={out_dir}")


def run_batch_siglip(args: argparse.Namespace) -> None:
    hidden_dim: int = args.hidden_dim
    topk: int = args.topk
    ckpt_path: str = args.ckpt
    chunk_size: int = args.chunk_size
    model = MODEL_SPECS["siglip"]
    dataset_spec = _dataset_spec(args.dataset, args.dataset_root)

    sae = _build_sae(model=model, hidden_dim=hidden_dim, topk=topk)

    ckpt_mgr = CheckpointManager(save_dir=BATCH_CKPT_MANAGER_SAVE_DIR)
    ckpt_mgr.load(path=ckpt_path, sae=sae)
    sae_analysis = SAEAnalysis(sae, DEFAULT_DEVICE)

    out_dir = _batch_out_dir("siglip", topk, hidden_dim, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_log(out_dir / "run.log")
    log(f"[INFO] SAE loaded: hidden_dim={hidden_dim}, sparsifier=topk({topk}), device=cuda")
    log(f"[INFO] Checkpoint: {ckpt_path}")
    log(f"[INFO] Dataset: name={dataset_spec.name}, root={dataset_spec.root}")

    data_mgr = _build_data_manager(dataset=dataset_spec, model=model, cache_dir=args.cache_dir)

    dataset = data_mgr.get_dataset()
    dataloader = data_mgr.get_dataloader(shuffle=False)
    log(f"[INFO] DataLoader prepared: size={len(dataset)}, batch_size={data_mgr.batch_size}, shuffle=False")

    t0 = time.perf_counter()
    stats = sae_analysis.activation_stats(dataloader, features_list=model.features_list)
    t1 = time.perf_counter()
    log(f"[INFO] activation_stats done in {t1 - t0:.2f}s")

    _log_activation_stats(log, stats)

    active_atom_ids = np.where(stats["global"]["count_active"] > 0)[0]
    log(f"[INFO] #active_atoms = {len(active_atom_ids)} / {hidden_dim}")

    _run_active_atom_overlay_chunks(
        sae_analysis=sae_analysis,
        dataloader=dataloader,
        active_atom_ids=active_atom_ids,
        chunk_size=chunk_size,
        out_dir=out_dir,
        log=log,
    )


def run_batch_dino(args: argparse.Namespace) -> None:
    hidden_dim: int = args.hidden_dim
    topk: int = args.topk
    ckpt_path: str = args.ckpt
    chunk_size: int = args.chunk_size
    model = MODEL_SPECS["dino"]
    dataset_spec = _dataset_spec(args.dataset, args.dataset_root)
    model_features_list = model.features_list

    if args.analysis_features_list is None:
        analysis_features_list = model_features_list
    else:
        analysis_features_list = _parse_int_list_csv(args.analysis_features_list)

    sae = _build_sae(model=model, hidden_dim=hidden_dim, topk=topk)

    ckpt_mgr = CheckpointManager(save_dir=BATCH_CKPT_MANAGER_SAVE_DIR)
    ckpt_mgr.load(path=ckpt_path, sae=sae)
    sae_analysis = SAEAnalysis(sae, DEFAULT_DEVICE)

    out_dir = _batch_out_dir("dino", topk, hidden_dim, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_log(out_dir / "run.log")
    log(f"[INFO] SAE loaded: hidden_dim={hidden_dim}, sparsifier=topk({topk}), device=cuda")
    log(f"[INFO] Checkpoint: {ckpt_path}")
    log(f"[INFO] Dataset: name={dataset_spec.name}, root={dataset_spec.root}")
    log(f"[INFO] ModelCfg.features_list={model_features_list}, analysis-features-list={analysis_features_list}")

    data_mgr = _build_data_manager(dataset=dataset_spec, model=model, cache_dir=args.cache_dir)

    dataset = data_mgr.get_dataset()
    dataloader = data_mgr.get_dataloader(shuffle=False)
    log(f"[INFO] DataLoader prepared: size={len(dataset)}, batch_size={data_mgr.batch_size}, shuffle=False")

    t0 = time.perf_counter()
    stats = sae_analysis.activation_stats(dataloader, features_list=analysis_features_list)
    t1 = time.perf_counter()
    log(f"[INFO] activation_stats done in {t1 - t0:.2f}s")

    _log_activation_stats(log, stats)

    active_atom_ids = np.where(stats["global"]["count_active"] > 0)[0]
    log(f"[INFO] #active_atoms = {len(active_atom_ids)} / {hidden_dim}")

    _run_active_atom_overlay_chunks(
        sae_analysis=sae_analysis,
        dataloader=dataloader,
        active_atom_ids=active_atom_ids,
        chunk_size=chunk_size,
        out_dir=out_dir,
        log=log,
    )


def main() -> None:
    args = parse_args()
    if args.mode == "individual":
        run_individual(args)
    elif args.mode == "batch_siglip":
        run_batch_siglip(args)
    elif args.mode == "batch_dino":
        run_batch_dino(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
