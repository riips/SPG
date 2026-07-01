"""Pluggable sinks for event persistence.

- EventSink: abstract protocol
- ParquetSink: rolling Parquet writer, gracefully disabled when pyarrow is unavailable
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np


class EventSink:
    """Interface for receiving activation events."""
    def on_events(
        self,
        sample: np.ndarray,
        token: np.ndarray,
        atom: np.ndarray,
        value: np.ndarray,
        meta: Dict[int, Dict[str, Any]] | None = None,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        return pa, pq
    except Exception as e:
        raise ImportError("ParquetSink を使うには `pyarrow` が必要です。`pip install pyarrow`") from e


class ParquetSink(EventSink):
    """Sink that writes rolling Parquet files.

    - Since append assumes a fixed schema, this implementation uses simple
      file rotation.
    - Metadata is projected from (sample_id -> dict) onto event rows and only
      the main keys are materialized as columns.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        prefix: str = "events",
        max_rows_per_file: int = 500_000,
        include_meta_keys: Optional[List[str]] = None,
    ) -> None:
        self.pa, self.pq = _require_pyarrow()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_rows_per_file = int(max(1, max_rows_per_file))
        self.include_meta_keys = include_meta_keys or ["cls", "global_id", "img_path", "anomaly", "specie_name"]

        self._rows: Dict[str, list] = {
            "sample": [],
            "token": [],
            "atom": [],
            "value": [],
        }
        for k in self.include_meta_keys:
            self._rows[f"meta_{k}"] = []

        self._file_index = 0
        self._row_count = 0

    def _flush(self) -> None:
        if self._row_count == 0:
            return
        table_cols = {k: self.pa.array(v) for k, v in self._rows.items()}
        table = self.pa.table(table_cols)
        out_path = self.out_dir / f"{self.prefix}-{self._file_index:05d}.parquet"
        self.pq.write_table(table, out_path)
        self._file_index += 1
        self._row_count = 0
        for k in list(self._rows.keys()):
            self._rows[k].clear()

    def on_events(
        self,
        sample: np.ndarray,
        token: np.ndarray,
        atom: np.ndarray,
        value: np.ndarray,
        meta: Dict[int, Dict[str, Any]] | None = None,
    ) -> None:
        n = int(sample.shape[0])
        # append
        self._rows["sample"].extend(sample.astype(np.int64).tolist())
        self._rows["token"].extend(token.astype(np.int32).tolist())
        self._rows["atom"].extend(atom.astype(np.int32).tolist())
        self._rows["value"].extend(value.astype(np.float32).tolist())

        if meta:
            # Project metadata by sample ID.
            for i in range(n):
                s = int(sample[i])
                m = meta.get(s, {}) if isinstance(meta, dict) else {}
                for k in self.include_meta_keys:
                    self._rows[f"meta_{k}"].append(m.get(k, None))
        else:
            for k in self.include_meta_keys:
                self._rows[f"meta_{k}"].extend([None] * n)

        self._row_count += n
        if self._row_count >= self.max_rows_per_file:
            self._flush()

    def close(self) -> None:
        self._flush()

