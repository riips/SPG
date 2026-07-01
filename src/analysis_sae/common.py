from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


def _ts() -> str:
    """Return a timestamp string in JST.

    Keep the behavior compatible with legacy scripts that used an identical
    fallback strategy when timezone construction fails.
    """

    try:
        jst = timezone(timedelta(hours=9))
        return datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S JST")
    except Exception:
        # Fallback to local time formatting (best-effort).
        return time.strftime("%Y-%m-%d %H:%M:%S")


def make_log(log_file: Path) -> Callable[[str], None]:
    """Create a logger function that prints and appends to `log_file`."""

    def log(msg: str) -> None:
        line = f"[{_ts()}] {msg}"
        print(line)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Continue even if file writing fails.
            pass

    return log


def load_sae_compat(ckpt_mgr: Any, ckpt_path: str, sae: Any) -> None:
    """Load SAE checkpoint in both single- and multi-SAE formats.

    Expected keys:
    - single: keys like "encoder.W.weight"
    - multi: keys like "sae_list.0.encoder.W.weight"
    """

    import torch

    states = torch.load(ckpt_path, map_location=ckpt_mgr.device)
    sd = states.get("sae_state_dict")
    if not isinstance(sd, dict):
        return

    sd_keys = list(sd.keys())
    if any(k.startswith("encoder.") or k.startswith("decoder.") for k in sd_keys):
        missing, unexpected = sae.load_state_dict(sd, strict=False)
    elif any(k.startswith("sae_list.0.") for k in sd_keys):
        remapped = {k.replace("sae_list.0.", "", 1): v for k, v in sd.items()}
        missing, unexpected = sae.load_state_dict(remapped, strict=False)
    else:
        # Unknown format; load directly to surface the mismatch.
        missing, unexpected = sae.load_state_dict(sd, strict=False)

    if missing or unexpected:
        import logging

        log = logging.getLogger(__name__)
        log.warning(
            "Component load_state_dict mismatch: missing=%s unexpected=%s",
            missing,
            unexpected,
        )

