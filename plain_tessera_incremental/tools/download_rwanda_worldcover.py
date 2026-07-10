from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectrajam.frame_sources import fetch_frame_sources, write_frame_sources_receipt


RWANDA_SOURCE_KEYS = (
    "world_bank_admin0",
    "worldcover_2021_s03e027",
    "worldcover_2021_s03e030",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned Rwanda boundary and WorldCover tiles"
    )
    parser.add_argument(
        "--source-root",
        default="/mnt/noobjam/rwanda_worldcover_mlp/sources",
        help="persistent destination for verified source files",
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser()
    fetched = fetch_frame_sources(
        source_root,
        keys=RWANDA_SOURCE_KEYS,
        max_attempts=args.max_attempts,
    )
    receipt = write_frame_sources_receipt(
        source_root / "sources.lock.json",
        fetched,
    )
    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "receipt": str(receipt),
                "sources": [
                    {
                        "key": item.source.key,
                        "path": str(item.path),
                        "bytes": item.path.stat().st_size,
                        "reused": item.reused,
                    }
                    for item in fetched
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
