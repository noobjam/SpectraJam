from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .windows import PrefixWindow, build_prefix_windows


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    input_parquet: Path
    output_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str | None
    windows: tuple[PrefixWindow, ...]
    source_crs: str
    pixel_size_m: int
    work_tile_m: int
    raster_chunk_pixels: int
    stac_endpoint: str
    s2_collection: str
    s1_collection: str
    stac_request_retries: int
    stac_query_halo_m: float
    stack_chunksize: int
    materialize_workers: int
    batch_size: int
    device: str
    wkt_column: str
    id_column: str
    label_column: str
    longitude_column: str
    latitude_column: str
    quadkey_column: str

    def validate(self, require_files: bool = False) -> None:
        if self.source_crs != "EPSG:4326":
            raise ValueError("this pipeline currently requires WKT in EPSG:4326")
        if self.pixel_size_m != 10:
            raise ValueError("plain TESSERA pixel inference requires a 10 m grid")
        if self.work_tile_m < 10_000 or self.work_tile_m > 20_000:
            raise ValueError("work_tile_m must be between 10 km and 20 km")
        if self.work_tile_m % self.pixel_size_m:
            raise ValueError("work_tile_m must be divisible by pixel_size_m")
        if self.raster_chunk_pixels < 32:
            raise ValueError("raster_chunk_pixels must be at least 32")
        if self.stac_endpoint != "https://planetarycomputer.microsoft.com/api/stac/v1":
            raise ValueError("the MPC checkpoint requires the pinned MPC STAC endpoint")
        if (self.s2_collection, self.s1_collection) != (
            "sentinel-2-l2a",
            "sentinel-1-rtc",
        ):
            raise ValueError("the MPC checkpoint requires MPC S2 L2A and S1 RTC collections")
        if not math.isfinite(self.stac_query_halo_m) or self.stac_query_halo_m < 0:
            raise ValueError("stac query_halo_m must be finite and non-negative")
        if (
            self.stac_request_retries < 0
            or self.stack_chunksize < 1
            or not 1 <= self.materialize_workers <= 64
            or self.batch_size < 1
        ):
            raise ValueError(
                "retry, stack chunk, materialize worker, and model batch settings are invalid"
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("checkpoint_sha256 must be null or a hexadecimal SHA-256")
        if self.windows[-1].end_exclusive.isoformat() != "2026-01-01":
            raise ValueError("the final cutoff must be 2026-01-01 to include 2025-12-31")
        if require_files:
            if not self.input_parquet.is_file():
                raise FileNotFoundError(f"ground-truth parquet not found: {self.input_parquet}")
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"TESSERA checkpoint not found: {self.checkpoint_path}")


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text())
    root = _mapping(raw, "config")
    input_data = _mapping(root.get("input"), "input")
    model = _mapping(root.get("model"), "model")
    temporal = _mapping(root.get("temporal"), "temporal")
    grid = _mapping(root.get("grid"), "grid")
    stac = _mapping(root.get("stac"), "stac")
    runtime = _mapping(root.get("runtime"), "runtime")
    columns = _mapping(root.get("columns"), "columns")

    checksum = model.get("checkpoint_sha256")
    config = PipelineConfig(
        input_parquet=_resolve_path(str(input_data["parquet"])),
        output_dir=_resolve_path(str(root["output_dir"])),
        checkpoint_path=_resolve_path(str(model["checkpoint_path"])),
        checkpoint_sha256=None if checksum in {None, ""} else str(checksum).lower(),
        windows=build_prefix_windows(
            str(temporal["start"]),
            [str(value) for value in temporal["cutoffs"]],
        ),
        source_crs=str(input_data.get("crs", "EPSG:4326")),
        pixel_size_m=int(grid.get("pixel_size_m", 10)),
        work_tile_m=int(grid.get("work_tile_m", 20_000)),
        raster_chunk_pixels=int(grid.get("raster_chunk_pixels", 128)),
        stac_endpoint=str(stac["endpoint"]),
        s2_collection=str(stac["collections"]["s2"]),
        s1_collection=str(stac["collections"]["s1"]),
        stac_request_retries=int(stac.get("request_retries", 3)),
        stac_query_halo_m=float(stac.get("query_halo_m", 500)),
        stack_chunksize=int(runtime.get("stack_chunksize", 256)),
        materialize_workers=int(runtime.get("materialize_workers", 8)),
        batch_size=int(runtime.get("batch_size", 256)),
        device=str(runtime.get("device", "auto")),
        wkt_column=str(columns.get("wkt", "wkt")),
        id_column=str(columns.get("id", "id")),
        label_column=str(columns.get("label", "landcover")),
        longitude_column=str(columns.get("longitude", "LONGITUDE")),
        latitude_column=str(columns.get("latitude", "LATITUDE")),
        quadkey_column=str(columns.get("quadkey", "QUADKEY")),
    )
    config.validate(require_files=False)
    return config
