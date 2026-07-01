#!/usr/bin/env python3
"""
Cache management utility

Usage:
    python cache_utils.py list          # Show cache entries
    python cache_utils.py clean         # Cache cleanup (dry run)
    python cache_utils.py clean --force # Cache cleanup (delete files)
"""

import argparse
import os
from pathlib import Path

from cache_manager import CacheManager


def main():
    parser = argparse.ArgumentParser(description="キャッシュ管理ユーティリティ")
    parser.add_argument("command", choices=["list", "clean"], help="実行するコマンド")
    parser.add_argument(
        "--force",
        action="store_true",
        help="実際にファイルを削除する（cleanコマンドのみ）",
    )
    default_cache_dir = os.environ.get("CACHE_DIR", str(Path.home() / ".cache" / "features"))
    parser.add_argument(
        "--cache-dir", default=default_cache_dir, help="キャッシュディレクトリのパス"
    )

    args = parser.parse_args()

    if args.command == "list":
        CacheManager.print_cache_info(args.cache_dir)
    elif args.command == "clean":
        CacheManager.clean_cache(args.cache_dir, dry_run=not args.force)


if __name__ == "__main__":
    main()
