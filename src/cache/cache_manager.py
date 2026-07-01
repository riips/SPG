from datetime import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, cast

from omegaconf import DictConfig, OmegaConf, ListConfig


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, DictConfig):
        return cast(Dict[str, Any], OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, dict):
        return cast(Dict[str, Any], obj)
    raise TypeError("key must be a dict-like object")


def _normalize_key(key: Any) -> Dict[str, Any]:
    # Convert DictConfig/dict into a regular dict.
    d = _as_dict(key)
    # Check required fields.
    required = ("model_id", "dataset_name", "image_size", "features_list")
    for k in required:
        if k not in d or d[k] is None:
            raise ValueError(f"missing required key field: {k}")
    # Normalize field types.
    model_id: str = str(d["model_id"])
    dataset_name: str = str(d["dataset_name"])
    image_size: int = int(d["image_size"])
    features_raw = d["features_list"]
    if isinstance(features_raw, ListConfig):
        features_raw = list(features_raw)
    if not isinstance(features_raw, (list, tuple)):
        raise TypeError(f"features_list must be a list or tuple of ints, got {type(features_raw)}")
    features_list = sorted({int(x) for x in features_raw})
    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "image_size": image_size,
        "features_list": features_list,
    }


def fingerprint_from_key(key: Dict[str, Any]) -> str:
    """Generate a fingerprint with the same procedure as the legacy implementation.

    Each element is JSON-encoded independently, concatenated, then SHA1-hashed
    and truncated to 8 characters.

    - Order: model_id -> dataset_name -> image_size -> features_list
    - Keep ensure_ascii at the default value (True), matching the legacy behavior.
    """
    m = hashlib.sha1()
    nk = _normalize_key(key)
    m.update(json.dumps(nk["model_id"], sort_keys=True).encode())
    m.update(json.dumps(nk["dataset_name"], sort_keys=True).encode())
    m.update(json.dumps(nk["image_size"], sort_keys=True).encode())
    m.update(json.dumps(nk["features_list"], sort_keys=True).encode())
    return m.hexdigest()[:8]


class CacheManager:
    def __init__(self, key: Dict[str, Any], cache_dir: str = os.path.expanduser("~/.cache/features")):
        """
        key: {"model_id": str, "dataset_name": str, "image_size": int, "features_list": List[int]}
        """
        self.cache_dir = cache_dir
        self.key = _normalize_key(key)
        self.fid = fingerprint_from_key(self.key)
        self.base_dir = os.path.join(cache_dir, self.fid)

        if os.path.exists(self.base_dir):
            self._ensure_meta_file()

    # ---------- API ----------
    def path_for(self, idx: int) -> str:
        sub = f"{idx:06d}"[:2]  # Split into two-digit subfolders.
        return os.path.join(self.base_dir, sub, f"{idx:06d}.pth")

    def ensure_cache_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        self._save_meta_file()

    def exists(self) -> bool:
        """
        Return whether the cache exists, meaning at least one .pth file is present.
        The directory and meta file are prepared here.
        """
        cache_path = self.base_dir

        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
            self._save_meta_file()
            return False

        meta_path = os.path.join(cache_path, "meta.json")
        if not os.path.exists(meta_path):
            self._save_meta_file()

        for _, _, files in os.walk(cache_path):
            if any(f.endswith(".pth") for f in files):
                return True
        return False

    def _save_meta_file(self) -> None:
        meta = {
            "model_id": self.key["model_id"],
            "dataset_name": self.key["dataset_name"],
            "image_size": self.key["image_size"],
            "features_list": self.key["features_list"],
            "fingerprint": self.fid,
            "created_at": datetime.now().isoformat(),
            "cache_dir": self.base_dir,
        }
        meta_path = os.path.join(self.base_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _ensure_meta_file(self) -> None:
        meta_path = os.path.join(self.base_dir, "meta.json")
        if not os.path.exists(meta_path):
            self._save_meta_file()
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._save_meta_file()
            return
        if "features_list" not in meta:
            meta["features_list"] = self.key["features_list"]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            return
        meta_features = meta.get("features_list")
        if isinstance(meta_features, list):
            if sorted({int(x) for x in meta_features}) != self.key["features_list"]:
                raise ValueError(
                    f"cache meta features_list mismatch: {meta_features} vs {self.key['features_list']}"
                )

    def get_meta_info(self):
        meta_path = os.path.join(self.base_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # ---------- Utils ----------
    @classmethod
    def list_caches(cls, cache_dir: str = os.path.expanduser("~/.cache/features")) -> List[Dict[str, Any]]:
        if not os.path.exists(cache_dir):
            return []

        caches: List[Dict[str, Any]] = []
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                meta_path = os.path.join(item_path, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        caches.append(
                            {
                                "fingerprint": item,
                                "model_id": meta.get("model_id", "Unknown"),
                                "dataset_name": meta.get("dataset_name", "Unknown"),
                                "image_size": meta.get("image_size", "Unknown"),
                                "features_list": meta.get("features_list", "Unknown"),
                                "created_at": meta.get("created_at", "Unknown"),
                                "cache_dir": item_path,
                            }
                        )
                    except (json.JSONDecodeError, KeyError):
                        caches.append(
                            {
                                "fingerprint": item,
                                "model_id": "Unknown (corrupted meta)",
                                "dataset_name": "Unknown (corrupted meta)",
                                "image_size": "Unknown (corrupted meta)",
                                "features_list": "Unknown (corrupted meta)",
                                "created_at": "Unknown",
                                "cache_dir": item_path,
                            }
                        )
        return caches

    @classmethod
    def print_cache_info(cls, cache_dir: str = os.path.expanduser("~/.cache/features")) -> None:
        caches: List[Dict[str, Any]] = cls.list_caches(cache_dir)
        if not caches:
            print("キャッシュが見つかりません。")
            return

        print(f"キャッシュディレクトリ: {cache_dir}")
        print("=" * 80)
        for cache in caches:
            print(f"フィンガープリント: {cache['fingerprint']}")
            print(f"モデルID: {cache['model_id']}")
            print(f"データセット名: {cache['dataset_name']}")
            print(f"画像サイズ: {cache['image_size']}")
            print(f"features_list: {cache['features_list']}")
            print(f"作成日時: {cache['created_at']}")
            print(f"ディレクトリ: {cache['cache_dir']}")
            print("-" * 40)

    @classmethod
    def clean_cache(cls, cache_dir: str = os.path.expanduser("~/.cache/features"), dry_run: bool = True) -> None:
        caches = cls.list_caches(cache_dir)
        if not caches:
            print("クリーンアップ対象のキャッシュがありません。")
            return

        print(f"キャッシュクリーンアップ {'(DRY RUN)' if dry_run else '(実際に削除)'}")
        print("=" * 60)

        for cache in caches:
            cache_dir_path = cache["cache_dir"]
            if os.path.exists(cache_dir_path):
                if dry_run:
                    print(
                        f"[DRY RUN] 削除予定: {cache['fingerprint']} ({cache['model_id']} + {cache['dataset_name']} @ {cache['image_size']}px)"
                    )
                else:
                    import shutil
                    shutil.rmtree(cache_dir_path)
                    print(
                        f"削除完了: {cache['fingerprint']} ({cache['model_id']} + {cache['dataset_name']} @ {cache['image_size']}px)"
                    )
