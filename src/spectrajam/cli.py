from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .config import load_config
from .contracts import sha256_file
from .ledger import AcquisitionLedger, IncompleteAcquisitionError
from .parity import verify_parity_fixture
from .sampling import (
    expand_years,
    read_candidates,
    read_manifest,
    select_candidates,
    stream_full_lattice,
    validate_candidate_extents,
    validate_manifest_universe,
    write_manifest,
)
from .stac import build_work_tiles


def _validate_config(args: argparse.Namespace) -> int:
    config = load_config(
        args.config,
        require_checkpoint=args.require_checkpoint or args.operational,
        require_boundaries=args.operational,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "operational": bool(args.operational),
                "name": config.name,
                "stage": config.stage,
            }
        )
    )
    return 0


def _sample(args: argparse.Namespace) -> int:
    config = load_config(args.config, require_boundaries=True)
    candidates = validate_candidate_extents(
        read_candidates(args.candidates),
        {country: policy.boundary_path for country, policy in config.extents.items()},
    )
    if config.sampling.mode == "full_lattice":
        points = stream_full_lattice(
            candidates,
            config.sampling.split_ratios,
            config.sampling.seed,
            config.sampling.min_distance_m,
        )
    else:
        points = select_candidates(
            candidates,
            points_per_country=config.sampling.points_per_country,
            min_per_stratum=config.sampling.min_per_stratum,
            allocation_power=config.sampling.allocation_power,
            split_ratios=config.sampling.split_ratios,
            seed=config.sampling.seed,
            min_distance_m=config.sampling.min_distance_m,
        )
    records = expand_years(
        points,
        {
            "train": config.years.train,
            "validation": config.years.validation,
            "test": config.years.test,
        },
    )
    record_count = write_manifest(args.output, records)
    print(
        json.dumps(
            {
                "selected_points": record_count
                // (
                    len(config.years.train)
                    + len(config.years.validation)
                    + len(config.years.test)
                ),
                "point_years": record_count,
                "output": args.output,
            }
        )
    )
    return 0


def _ledger_init(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    records = validate_manifest_universe(
        read_manifest(args.manifest),
        config.countries,
        {
            "train": config.years.train,
            "validation": config.years.validation,
            "test": config.years.test,
        },
        config.sampling.points_per_country,
    )
    ledger = AcquisitionLedger(args.database)
    inserted = ledger.bootstrap(
        records,
        args.modalities,
        config.retry.max_attempts,
        manifest_sha256=sha256_file(args.manifest),
        config_sha256=sha256_file(args.config),
    )
    print(json.dumps({"inserted": inserted, "summary": ledger.summary()}))
    return 0


def _ledger_status(args: argparse.Namespace) -> int:
    ledger = AcquisitionLedger(args.database)
    print(json.dumps({"summary": ledger.summary(), "failures": ledger.failures(args.limit)}))
    return 0


def _ledger_assert(args: argparse.Namespace) -> int:
    ledger = AcquisitionLedger(args.database)
    try:
        ledger.assert_complete()
    except IncompleteAcquisitionError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"complete": True, "summary": ledger.summary()}))
    return 0


def _catalog_plan(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    tiles = build_work_tiles(records)
    print(json.dumps([asdict(tile) | {"key": tile.key} for tile in tiles], indent=2))
    return 0


def _verify_parity(args: argparse.Namespace) -> int:
    config = load_config(args.config, require_checkpoint=True)
    receipt = verify_parity_fixture(
        config.base_model.checkpoint_path,
        config.base_model.checkpoint_sha256,
        args.fixture,
        args.receipt,
        args.atol,
    )
    print(json.dumps(receipt, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrajam")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.add_argument("--require-checkpoint", action="store_true")
    validate.add_argument(
        "--operational",
        action="store_true",
        help="also verify boundary/checkpoint files and their SHA-256 digests",
    )
    validate.set_defaults(handler=_validate_config)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--config", required=True)
    sample.add_argument("--candidates", required=True)
    sample.add_argument("--output", required=True)
    sample.set_defaults(handler=_sample)

    ledger_init = subparsers.add_parser("ledger-init")
    ledger_init.add_argument("--config", required=True)
    ledger_init.add_argument("--manifest", required=True)
    ledger_init.add_argument("--database", required=True)
    ledger_init.add_argument("--modalities", nargs="+", default=["s1", "s2"])
    ledger_init.set_defaults(handler=_ledger_init)

    ledger_status = subparsers.add_parser("ledger-status")
    ledger_status.add_argument("--database", required=True)
    ledger_status.add_argument("--limit", type=int, default=20)
    ledger_status.set_defaults(handler=_ledger_status)

    ledger_assert = subparsers.add_parser("ledger-assert-complete")
    ledger_assert.add_argument("--database", required=True)
    ledger_assert.set_defaults(handler=_ledger_assert)

    catalog_plan = subparsers.add_parser("catalog-plan")
    catalog_plan.add_argument("--manifest", required=True)
    catalog_plan.set_defaults(handler=_catalog_plan)

    parity = subparsers.add_parser("verify-upstream-parity")
    parity.add_argument("--config", required=True)
    parity.add_argument("--fixture", required=True)
    parity.add_argument("--receipt", required=True)
    parity.add_argument("--atol", type=float, default=1e-5)
    parity.set_defaults(handler=_verify_parity)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
