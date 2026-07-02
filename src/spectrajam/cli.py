from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import TESSERA_V11_MPC_ENCODER, fetch_verified_artifact
from .config import load_config
from .contracts import ContractError, sha256_file
from .frame_sources import (
    FRAME_SOURCES,
    fetch_frame_sources,
    frame_operation_lock,
    write_frame_sources_receipt,
)
from .ledger import AcquisitionLedger, IncompleteAcquisitionError
from .sampling import (
    PROJECTED_LATTICE_DISTANCE_TOLERANCE,
    expand_years,
    read_candidates,
    read_manifest,
    select_candidates,
    stream_full_lattice,
    validate_candidate_extents,
    validate_manifest_universe,
    verify_sampling_receipt,
    write_manifest,
)
from .stac import (
    ProviderProfile,
    STACCatalog,
    build_work_tiles,
    discover_catalogs,
)


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


def _sample_unlocked(args: argparse.Namespace) -> int:
    from . import candidate_frame as candidate_frame_module
    from . import frame_sources as frame_sources_module
    from .candidate_frame import (
        candidate_frame_contract,
        verify_candidate_frame_receipt,
        write_json_atomic,
    )

    config = load_config(args.config, require_boundaries=True)
    candidate_receipt = verify_candidate_frame_receipt(
        args.candidates, args.candidate_receipt
    )
    if candidate_receipt.get("frame_contract") != candidate_frame_contract(config):
        raise ContractError(
            "candidate-frame receipt does not match the experiment frame contract"
        )
    expected_implementation = {
        "candidate_frame_sha256": sha256_file(candidate_frame_module.__file__),
        "frame_sources_sha256": sha256_file(frame_sources_module.__file__),
    }
    recorded_implementation = candidate_receipt.get("implementation", {})
    if any(
        recorded_implementation.get(key) != value
        for key, value in expected_implementation.items()
    ):
        raise ContractError(
            "candidate-frame receipt was produced by a different implementation"
        )
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
            PROJECTED_LATTICE_DISTANCE_TOLERANCE,
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
            min_distance_relative_tolerance=PROJECTED_LATTICE_DISTANCE_TOLERANCE,
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
    sampling_receipt = (
        Path(args.receipt)
        if args.receipt
        else Path(args.output).with_suffix(Path(args.output).suffix + ".receipt.json")
    )
    write_json_atomic(
        sampling_receipt,
        {
            "schema": "spectrajam-sampling-v1",
            "config_sha256": sha256_file(args.config),
            "candidate_frame_receipt_sha256": sha256_file(args.candidate_receipt),
            "candidate_csv_sha256": candidate_receipt["candidate_output"]["sha256"],
            "manifest": {
                "path": args.output,
                "bytes": Path(args.output).stat().st_size,
                "sha256": sha256_file(args.output),
                "point_years": record_count,
            },
        },
    )
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
                "receipt": str(sampling_receipt),
            }
        )
    )
    return 0


def _sample(args: argparse.Namespace) -> int:
    with frame_operation_lock(Path(args.output).parent, "sampling"):
        return _sample_unlocked(args)


def _ledger_init(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    verify_sampling_receipt(args.manifest, args.config, args.sampling_receipt)
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
        sampling_receipt_sha256=sha256_file(args.sampling_receipt),
    )
    print(json.dumps({"inserted": inserted, "summary": ledger.summary()}))
    return 0


def _ledger_status(args: argparse.Namespace) -> int:
    ledger = AcquisitionLedger(args.database)
    print(
        json.dumps(
            {
                "summary": ledger.summary(),
                "outcomes": ledger.outcomes(),
                "failures": ledger.failures(args.limit),
            }
        )
    )
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


def _catalog_discover(args: argparse.Namespace) -> int:
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
    profile = ProviderProfile(
        name=config.stac.profile,
        endpoint=config.stac.endpoint,
        collections=config.stac.collections,
        checkpoint_source=config.base_model.data_source,
    )
    catalog = STACCatalog(
        profile,
        request_retries=max(0, config.retry.max_attempts - 1),
        backoff_factor=config.retry.base_delay_seconds,
        backoff_max=config.retry.max_delay_seconds,
    )
    results = discover_catalogs(records, args.output, catalog, args.modalities)
    print(
        json.dumps(
            {
                "queries": len(results),
                "reused": sum(result.reused for result in results),
                "items_referenced": sum(result.item_count for result in results),
                "item_documents_written": sum(
                    result.item_documents_written for result in results
                ),
                "output": args.output,
            }
        )
    )
    return 0


def _materialize(args: argparse.Namespace) -> int:
    from .materialize import (
        default_worker_id,
        json_progress,
        materializer_contract_sha256,
        preflight_materialization,
        run_materialization,
    )

    if args.max_groups is not None and args.max_groups < 1:
        raise ContractError("--max-groups must be positive")
    if args.asset_attempts is not None and args.asset_attempts < 1:
        raise ContractError("--asset-attempts must be positive")
    config = load_config(args.config)
    if config.stage != "smoke":
        raise ContractError(
            "the v1 per-point materializer is smoke-only; pilot/full require "
            "batched Parquet shards to avoid millions of tiny files"
        )
    verify_sampling_receipt(args.manifest, args.config, args.sampling_receipt)
    records = list(
        validate_manifest_universe(
            read_manifest(args.manifest),
            config.countries,
            {
                "train": config.years.train,
                "validation": config.years.validation,
                "test": config.years.test,
            },
            config.sampling.points_per_country,
        )
    )
    profile = ProviderProfile(
        name=config.stac.profile,
        endpoint=config.stac.endpoint,
        collections=config.stac.collections,
        checkpoint_source=config.base_model.data_source,
    )
    tiles = {
        (tile.country, tile.spatial_block, tile.year): tile for tile in build_work_tiles(records)
    }
    ledger = AcquisitionLedger(args.database)
    ledger.require_binding(
        manifest_sha256=sha256_file(args.manifest),
        config_sha256=sha256_file(args.config),
        sampling_receipt_sha256=sha256_file(args.sampling_receipt),
    )
    print(
        json.dumps({"event": "materialization-preflight", "work_tiles": len(tiles)}),
        flush=True,
    )
    catalog_inventory_sha256 = preflight_materialization(
        tile_by_identity=tiles,
        catalog_root=args.catalog_root,
        output_root=args.output,
        profile=profile,
        modalities=ledger.modalities(),
        verify_remote_assets=True,
        max_attempts=args.asset_attempts or config.retry.max_attempts,
        base_delay_seconds=config.retry.base_delay_seconds,
        max_delay_seconds=config.retry.max_delay_seconds,
    )
    implementation_sha256 = materializer_contract_sha256()
    ledger.bind_metadata("catalog_inventory_sha256", catalog_inventory_sha256)
    ledger.bind_metadata("materializer_contract_sha256", implementation_sha256)
    print(
        json.dumps(
            {
                "event": "materialization-preflight-complete",
                "catalog_inventory_sha256": catalog_inventory_sha256,
                "materializer_contract_sha256": implementation_sha256,
            }
        ),
        flush=True,
    )
    result = run_materialization(
        ledger=ledger,
        tile_by_identity=tiles,
        catalog_root=args.catalog_root,
        output_root=args.output,
        profile=profile,
        worker_id=args.worker_id or default_worker_id(),
        batch_points=args.batch_points,
        lease_seconds=config.retry.lease_seconds,
        max_attempts=args.asset_attempts or config.retry.max_attempts,
        base_delay_seconds=config.retry.base_delay_seconds,
        max_delay_seconds=config.retry.max_delay_seconds,
        max_groups=args.max_groups,
        progress=json_progress,
        implementation_sha256=implementation_sha256,
    )
    print(
        json.dumps(
            {
                "result": asdict(result),
                "summary": ledger.summary(),
                "outcomes": ledger.outcomes(),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    summary = ledger.summary()
    if summary["failed"]:
        return 2
    if any(summary[state] for state in ("pending", "retry", "running")):
        return 3
    return 0


def _verify_parity(args: argparse.Namespace) -> int:
    from .parity import verify_parity_fixture

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


def _fetch_checkpoint(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    artifact = TESSERA_V11_MPC_ENCODER
    if config.base_model.data_source != "mpc" or config.base_model.version != "1.1":
        raise ContractError("the pinned fetch command supports only TESSERA v1.1 MPC")
    if config.base_model.checkpoint_sha256.lower() != artifact.sha256:
        raise ContractError(
            "config checkpoint_sha256 does not match the pinned TESSERA v1.1 MPC artifact"
        )
    destination = Path(config.base_model.checkpoint_path)
    reused = destination.is_file()
    fetched = fetch_verified_artifact(
        artifact,
        destination,
        max_attempts=args.max_attempts,
    )
    print(
        json.dumps(
            {
                "path": str(fetched),
                "bytes": fetched.stat().st_size,
                "sha256": artifact.sha256,
                "source_revision": artifact.revision,
                "reused": reused,
            }
        )
    )
    return 0


def _model_smoke(args: argparse.Namespace) -> int:
    from .model_smoke import run_model_smoke

    config = load_config(args.config, require_checkpoint=True)
    report = run_model_smoke(
        config.base_model.checkpoint_path,
        config.base_model.checkpoint_sha256,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _fetch_frame_sources(args: argparse.Namespace) -> int:
    fetched = fetch_frame_sources(
        args.source_root,
        max_attempts=args.max_attempts,
    )
    receipt = Path(args.receipt) if args.receipt else Path(args.source_root) / "sources.lock.json"
    write_frame_sources_receipt(receipt, fetched)
    print(
        json.dumps(
            {
                "sources": len(fetched),
                "bytes": sum(item.path.stat().st_size for item in fetched),
                "reused": sum(item.reused for item in fetched),
                "source_root": args.source_root,
                "receipt": str(receipt),
            }
        )
    )
    return 0


def _candidate_frame(args: argparse.Namespace) -> int:
    from .candidate_frame import build_candidate_frame

    fetched = fetch_frame_sources(
        args.source_root,
        max_attempts=args.max_attempts,
    )
    source_receipt = Path(args.source_root) / "sources.lock.json"
    write_frame_sources_receipt(source_receipt, fetched)
    config = load_config(args.config, require_boundaries=True)
    expected_boundary = FRAME_SOURCES["world_bank_admin0"].destination(args.source_root)
    configured_boundaries = {
        Path(policy.boundary_path) for policy in config.extents.values()
    }
    if configured_boundaries != {expected_boundary}:
        raise ContractError(
            f"extent_policy must point to the fetched ADM0 artifact {expected_boundary}"
        )
    result = build_candidate_frame(
        config,
        args.source_root,
        args.output,
        args.receipt,
        chunk_size=args.chunk_size,
    )
    print(
        json.dumps(
            {
                "candidates": result.candidates_by_country,
                "exclusions": result.exclusions_by_country,
                "output": str(result.output),
                "sha256": result.output_sha256,
                "receipt": str(result.receipt),
            },
            sort_keys=True,
        )
    )
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
    sample.add_argument("--candidate-receipt", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--receipt")
    sample.set_defaults(handler=_sample)

    ledger_init = subparsers.add_parser("ledger-init")
    ledger_init.add_argument("--config", required=True)
    ledger_init.add_argument("--manifest", required=True)
    ledger_init.add_argument("--sampling-receipt", required=True)
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

    catalog_discover = subparsers.add_parser("catalog-discover")
    catalog_discover.add_argument("--config", required=True)
    catalog_discover.add_argument("--manifest", required=True)
    catalog_discover.add_argument("--output", required=True)
    catalog_discover.add_argument("--modalities", nargs="+", default=["s1", "s2"])
    catalog_discover.set_defaults(handler=_catalog_discover)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", required=True)
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--sampling-receipt", required=True)
    materialize.add_argument("--catalog-root", required=True)
    materialize.add_argument("--database", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--worker-id")
    materialize.add_argument("--batch-points", type=int, default=256)
    materialize.add_argument("--asset-attempts", type=int)
    materialize.add_argument("--max-groups", type=int)
    materialize.set_defaults(handler=_materialize)

    parity = subparsers.add_parser("verify-upstream-parity")
    parity.add_argument("--config", required=True)
    parity.add_argument("--fixture", required=True)
    parity.add_argument("--receipt", required=True)
    parity.add_argument("--atol", type=float, default=1e-5)
    parity.set_defaults(handler=_verify_parity)

    fetch_checkpoint = subparsers.add_parser("fetch-checkpoint")
    fetch_checkpoint.add_argument("--config", required=True)
    fetch_checkpoint.add_argument("--max-attempts", type=int, default=5)
    fetch_checkpoint.set_defaults(handler=_fetch_checkpoint)

    model_smoke = subparsers.add_parser("model-smoke")
    model_smoke.add_argument("--config", required=True)
    model_smoke.add_argument("--device", default="cuda:0")
    model_smoke.add_argument("--seed", type=int, default=20260701)
    model_smoke.add_argument("--batch-size", type=int, default=4)
    model_smoke.set_defaults(handler=_model_smoke)

    frame_sources = subparsers.add_parser("fetch-frame-sources")
    frame_sources.add_argument("--source-root", default="data/frame")
    frame_sources.add_argument("--receipt")
    frame_sources.add_argument("--max-attempts", type=int, default=5)
    frame_sources.set_defaults(handler=_fetch_frame_sources)

    candidate_frame = subparsers.add_parser("candidate-frame")
    candidate_frame.add_argument("--config", required=True)
    candidate_frame.add_argument("--source-root", default="data/frame")
    candidate_frame.add_argument("--output", required=True)
    candidate_frame.add_argument("--receipt", required=True)
    candidate_frame.add_argument("--max-attempts", type=int, default=5)
    candidate_frame.add_argument("--chunk-size", type=int, default=50_000)
    candidate_frame.set_defaults(handler=_candidate_frame)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
