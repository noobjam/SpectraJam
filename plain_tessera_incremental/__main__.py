from __future__ import annotations

import argparse
import json
import logging

from .config import load_config
from .pipeline import preflight, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate four cumulative plain-TESSERA pixel embeddings for WKT fields"
    )
    parser.add_argument(
        "--config",
        default="plain_tessera_incremental/config.yaml",
        help="pipeline YAML (default: plain_tessera_incremental/config.yaml)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate paths, columns, checksum, and windows without STAC access",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    result = preflight(config) if args.preflight_only else run_pipeline(config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
