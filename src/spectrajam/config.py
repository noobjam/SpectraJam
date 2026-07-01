from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import (
    CANONICAL_S2_BANDS,
    TESSERA_UPSTREAM_COMMIT,
    ContractError,
    require_band_order,
    require_sha256,
)


def _tuple_of_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{name} must be a non-empty list")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{name} contains duplicate years")
    return result


def _require_exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


@dataclass(frozen=True, slots=True)
class YearSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def validate(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ContractError("train, validation, and test years must be disjoint")
        for year in self.train + self.validation + self.test:
            if not 2017 <= year <= 2025:
                raise ContractError(f"year {year} is outside the pinned experiment window")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    mode: str
    points_per_country: int | None
    lattice_spacing_m: float
    min_per_stratum: int
    allocation_power: float
    seed: int
    spatial_block_km: float
    min_distance_m: float
    split_ratios: dict[str, float]

    def validate(self) -> None:
        if self.mode not in {"stratified_subset", "full_lattice"}:
            raise ContractError("sampling.mode must be stratified_subset or full_lattice")
        if self.mode == "stratified_subset" and (
            self.points_per_country is None or self.points_per_country <= 0
        ):
            raise ContractError("stratified_subset requires positive points_per_country")
        if self.mode == "full_lattice" and self.points_per_country is not None:
            raise ContractError("full_lattice must leave points_per_country null")
        if self.lattice_spacing_m < 10:
            raise ContractError("lattice_spacing_m must be at least one Sentinel-2 pixel")
        if self.min_per_stratum < 0:
            raise ContractError("min_per_stratum cannot be negative")
        if not 0 <= self.allocation_power <= 1:
            raise ContractError("allocation_power must be in [0, 1]")
        if self.spatial_block_km < 10:
            raise ContractError("spatial blocks below 10 km invite spatial leakage")
        if self.min_distance_m < 10:
            raise ContractError("min_distance_m must be at least one Sentinel-2 pixel")
        if set(self.split_ratios) != {"train", "validation", "test"}:
            raise ContractError("split_ratios must define train, validation, and test")
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-9:
            raise ContractError("split_ratios must sum to 1")
        if any(value <= 0 for value in self.split_ratios.values()):
            raise ContractError("all split ratios must be positive")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    lease_seconds: int

    def validate(self) -> None:
        if self.max_attempts < 1:
            raise ContractError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ContractError("base_delay_seconds cannot be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ContractError("max_delay_seconds must be >= base_delay_seconds")
        if self.lease_seconds < 30:
            raise ContractError("lease_seconds must be at least 30")


@dataclass(frozen=True, slots=True)
class STACConfig:
    profile: str
    endpoint: str
    collections: dict[str, str]
    work_tile_km: float
    item_cloud_filter: float | None
    retry: RetryConfig

    def validate(self, data_source: str) -> None:
        if self.profile != "mpc-v1.1":
            raise ContractError("the initial implementation supports only mpc-v1.1")
        if data_source != "mpc":
            raise ContractError("mpc-v1.1 STAC must use the MPC checkpoint and statistics")
        if self.endpoint != "https://planetarycomputer.microsoft.com/api/stac/v1":
            raise ContractError("mpc-v1.1 endpoint does not match the pinned provider profile")
        if self.collections != {"s2": "sentinel-2-l2a", "s1": "sentinel-1-rtc"}:
            raise ContractError("mpc-v1.1 collections do not match the pinned provider profile")
        if not 10 <= self.work_tile_km <= 20:
            raise ContractError("STAC work tiles must be between 10 and 20 km")
        if self.item_cloud_filter is not None:
            raise ContractError("item_cloud_filter must remain null; cloud validity is per pixel")
        self.retry.validate()


@dataclass(frozen=True, slots=True)
class BoundaryPolicy:
    source: str
    boundary_path: str
    boundary_sha256: str
    ndlsa: str

    def validate(self, require_file: bool = False) -> None:
        if self.source != "world_bank_official_boundaries_v2":
            raise ContractError("boundary source must be world_bank_official_boundaries_v2")
        if self.ndlsa != "exclude":
            raise ContractError("the default reproducible extent policy excludes NDLSA")
        if require_file:
            path = Path(self.boundary_path).expanduser()
            if not path.is_file():
                raise ContractError(f"boundary not found: {path}")
            require_sha256(path, self.boundary_sha256)


@dataclass(frozen=True, slots=True)
class StudentConfig:
    model_dim: int
    layers: int
    heads: int
    feedforward_dim: int
    output_dim: int
    temporal_view_sizes: tuple[int, ...]

    def validate(self) -> None:
        if min(self.model_dim, self.layers, self.heads, self.feedforward_dim) < 1:
            raise ContractError("student architecture values must be positive")
        if self.model_dim % self.heads:
            raise ContractError("student model_dim must be divisible by heads")
        if self.output_dim != 128:
            raise ContractError("student output_dim must remain 128 for the first experiment")
        if not self.temporal_view_sizes or any(value < 1 for value in self.temporal_view_sizes):
            raise ContractError("student temporal_view_sizes must be positive")


@dataclass(frozen=True, slots=True)
class LoRAExperimentConfig:
    ranks: tuple[int, ...]
    targets: tuple[str, ...]
    base_anchor_weights: tuple[float, ...]
    window_conditioner: bool

    def validate(self) -> None:
        if not self.ranks or any(value < 1 for value in self.ranks):
            raise ContractError("LoRA ranks must be positive")
        if not set(self.targets) <= {"attention", "attention_ffn"} or not self.targets:
            raise ContractError("unsupported LoRA target sweep")
        if not self.base_anchor_weights or any(value < 0 for value in self.base_anchor_weights):
            raise ContractError("LoRA base anchor weights must be non-negative")
        if not self.window_conditioner:
            raise ContractError(
                "arbitrary-window LoRA requires the explicit window conditioner"
            )


@dataclass(frozen=True, slots=True)
class TemporalWindowsConfig:
    boundary: str
    minimum_days: int
    maximum_days: int
    anchor_days: tuple[int, ...]
    anchor_probability: float
    prefix_probability: float
    view_dropout_probability: float
    evaluation_days: tuple[int, ...]
    cross_year_policy: str
    teacher_target: str
    teacher_anchor_full_weight_days: int
    teacher_anchor_full_observations: int

    def validate(self) -> None:
        if self.boundary != "half_open":
            raise ContractError("temporal windows must use half-open [start, end) bounds")
        if self.minimum_days < 1 or self.maximum_days < self.minimum_days:
            raise ContractError("invalid temporal-window duration range")
        for name, values in (
            ("anchor_days", self.anchor_days),
            ("evaluation_days", self.evaluation_days),
        ):
            if not values or len(set(values)) != len(values):
                raise ContractError(f"temporal_windows.{name} must be non-empty and unique")
            if any(value < self.minimum_days or value > self.maximum_days for value in values):
                raise ContractError(
                    f"temporal_windows.{name} must lie inside the duration range"
                )
            if not {7, 14} <= set(values):
                raise ContractError(f"temporal_windows.{name} must include 7 and 14 days")
        if not 0 <= self.anchor_probability <= 1:
            raise ContractError("temporal window anchor_probability must be in [0, 1]")
        if not 0 <= self.prefix_probability <= 1:
            raise ContractError("temporal window prefix_probability must be in [0, 1]")
        if not 0 <= self.view_dropout_probability <= 1:
            raise ContractError("temporal window view_dropout_probability must be in [0, 1]")
        if self.teacher_target != "same_window":
            raise ContractError(
                "windowed training must use same_window teacher targets to prevent leakage"
            )
        if self.cross_year_policy != "same_temporal_split":
            raise ContractError(
                "cross-year windows must remain inside one temporal split"
            )
        if self.teacher_anchor_full_weight_days < 1:
            raise ContractError("teacher_anchor_full_weight_days must be positive")
        if self.teacher_anchor_full_observations < 1:
            raise ContractError("teacher_anchor_full_observations must be positive")

    def sampling_policy(self):
        from .windows import WindowSamplingPolicy

        return WindowSamplingPolicy(
            minimum_days=self.minimum_days,
            maximum_days=self.maximum_days,
            anchor_days=self.anchor_days,
            anchor_probability=self.anchor_probability,
            prefix_probability=self.prefix_probability,
            view_dropout_probability=self.view_dropout_probability,
        )


@dataclass(frozen=True, slots=True)
class StrataConfig:
    primary: tuple[str, ...]
    balancing: tuple[str, ...]

    def validate(self) -> None:
        if self.primary != ("resolve_ecoregion", "esa_worldcover_2021"):
            raise ContractError("primary strata must be RESOLVE ecoregion × WorldCover 2021")
        required = {"elevation_quantile", "climate_quantile", "valid_observation_quantile"}
        if not required <= set(self.balancing):
            missing = sorted(required - set(self.balancing))
            raise ContractError(f"strata balancing is missing {missing}")


@dataclass(frozen=True, slots=True)
class BaseModelConfig:
    version: str
    data_source: str
    upstream_commit: str
    checkpoint_path: str
    checkpoint_sha256: str
    s2_bands: tuple[str, ...]

    def validate(self, require_checkpoint: bool = False) -> None:
        if self.version != "1.1":
            raise ContractError("the first SpectraJam experiments are pinned to TESSERA v1.1")
        if self.data_source not in {"mpc", "aws"}:
            raise ContractError("data_source must be mpc or aws")
        if self.upstream_commit != TESSERA_UPSTREAM_COMMIT:
            raise ContractError(
                f"upstream_commit must remain pinned to {TESSERA_UPSTREAM_COMMIT}"
            )
        require_band_order(self.s2_bands)
        if require_checkpoint:
            path = Path(self.checkpoint_path).expanduser()
            if not path.is_file():
                raise ContractError(f"checkpoint not found: {path}")
            require_sha256(path, self.checkpoint_sha256)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    stage: str
    countries: tuple[str, ...]
    years: YearSplit
    sampling: SamplingConfig
    stac: STACConfig
    extents: dict[str, BoundaryPolicy]
    strata: StrataConfig
    base_model: BaseModelConfig
    temporal_windows: TemporalWindowsConfig
    student: StudentConfig
    lora: LoRAExperimentConfig
    raw: dict[str, Any]

    @property
    def retry(self) -> RetryConfig:
        return self.stac.retry

    def validate(
        self, require_checkpoint: bool = False, require_boundaries: bool = False
    ) -> None:
        if not self.name:
            raise ContractError("name is required")
        if self.stage not in {"smoke", "pilot", "full"}:
            raise ContractError("stage must be smoke, pilot, or full")
        if set(self.countries) != {"RWA", "ISR"}:
            raise ContractError("the initial experiment must include RWA and ISR")
        self.years.validate()
        self.sampling.validate()
        self.base_model.validate(require_checkpoint=require_checkpoint)
        self.stac.validate(self.base_model.data_source)
        if set(self.extents) != set(self.countries):
            raise ContractError("extent_policy must define exactly the configured countries")
        for policy in self.extents.values():
            policy.validate(require_file=require_boundaries)
        self.strata.validate()
        self.temporal_windows.validate()
        self.student.validate()
        self.lora.validate()


def load_config(
    path: str | Path,
    require_checkpoint: bool = False,
    require_boundaries: bool = False,
) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ContractError("configuration root must be a mapping")

    _require_exact_keys(
        data,
        {
            "name",
            "stage",
            "countries",
            "years",
            "sampling",
            "extent_policy",
            "strata",
            "stac",
            "base_model",
            "temporal_windows",
            "student",
            "lora",
        },
        "configuration",
    )

    years_data = data["years"]
    sampling_data = data["sampling"]
    retry_data = data["stac"]["retry"]
    stac_data = data["stac"]
    model_data = data["base_model"]
    student_data = data["student"]
    lora_data = data["lora"]
    strata_data = data["strata"]
    window_data = data["temporal_windows"]

    _require_exact_keys(years_data, {"train", "validation", "test"}, "years")
    _require_exact_keys(
        sampling_data,
        {
            "mode",
            "points_per_country",
            "lattice_spacing_m",
            "min_per_stratum",
            "allocation_power",
            "seed",
            "spatial_block_km",
            "min_distance_m",
            "split_ratios",
        },
        "sampling",
    )
    _require_exact_keys(
        sampling_data["split_ratios"],
        {"train", "validation", "test"},
        "sampling.split_ratios",
    )
    _require_exact_keys(
        stac_data,
        {
            "profile",
            "endpoint",
            "collections",
            "work_tile_km",
            "item_cloud_filter",
            "retry",
        },
        "stac",
    )
    _require_exact_keys(
        retry_data,
        {"max_attempts", "base_delay_seconds", "max_delay_seconds", "lease_seconds"},
        "stac.retry",
    )
    _require_exact_keys(strata_data, {"primary", "balancing"}, "strata")
    _require_exact_keys(
        model_data,
        {
            "version",
            "data_source",
            "upstream_commit",
            "checkpoint_path",
            "checkpoint_sha256",
            "s2_bands",
        },
        "base_model",
    )
    _require_exact_keys(
        window_data,
        {
            "boundary",
            "minimum_days",
            "maximum_days",
            "anchor_days",
            "anchor_probability",
            "prefix_probability",
            "view_dropout_probability",
            "evaluation_days",
            "cross_year_policy",
            "teacher_target",
            "teacher_anchor_full_weight_days",
            "teacher_anchor_full_observations",
        },
        "temporal_windows",
    )
    _require_exact_keys(
        student_data,
        {
            "model_dim",
            "layers",
            "heads",
            "feedforward_dim",
            "output_dim",
            "temporal_view_sizes",
        },
        "student",
    )
    _require_exact_keys(
        lora_data,
        {"ranks", "targets", "base_anchor_weights", "window_conditioner"},
        "lora",
    )
    for country, policy in data["extent_policy"].items():
        _require_exact_keys(
            policy,
            {"source", "boundary_path", "boundary_sha256", "ndlsa"},
            f"extent_policy.{country}",
        )

    config = ExperimentConfig(
        name=str(data["name"]),
        stage=str(data["stage"]),
        countries=tuple(str(item) for item in data["countries"]),
        years=YearSplit(
            train=_tuple_of_ints(years_data["train"], "years.train"),
            validation=_tuple_of_ints(years_data["validation"], "years.validation"),
            test=_tuple_of_ints(years_data["test"], "years.test"),
        ),
        sampling=SamplingConfig(
            mode=str(sampling_data["mode"]),
            points_per_country=(
                None
                if sampling_data.get("points_per_country") is None
                else int(sampling_data["points_per_country"])
            ),
            lattice_spacing_m=float(sampling_data["lattice_spacing_m"]),
            min_per_stratum=int(sampling_data["min_per_stratum"]),
            allocation_power=float(sampling_data["allocation_power"]),
            seed=int(sampling_data["seed"]),
            spatial_block_km=float(sampling_data["spatial_block_km"]),
            min_distance_m=float(sampling_data["min_distance_m"]),
            split_ratios={k: float(v) for k, v in sampling_data["split_ratios"].items()},
        ),
        stac=STACConfig(
            profile=str(stac_data["profile"]),
            endpoint=str(stac_data["endpoint"]),
            collections={k: str(v) for k, v in stac_data["collections"].items()},
            work_tile_km=float(stac_data["work_tile_km"]),
            item_cloud_filter=(
                None
                if stac_data.get("item_cloud_filter") is None
                else float(stac_data["item_cloud_filter"])
            ),
            retry=RetryConfig(
                max_attempts=int(retry_data["max_attempts"]),
                base_delay_seconds=float(retry_data["base_delay_seconds"]),
                max_delay_seconds=float(retry_data["max_delay_seconds"]),
                lease_seconds=int(retry_data["lease_seconds"]),
            ),
        ),
        extents={
            code: BoundaryPolicy(
                source=str(policy["source"]),
                boundary_path=str(policy["boundary_path"]),
                boundary_sha256=str(policy["boundary_sha256"]),
                ndlsa=str(policy["ndlsa"]),
            )
            for code, policy in data["extent_policy"].items()
        },
        strata=StrataConfig(
            primary=tuple(str(value) for value in strata_data["primary"]),
            balancing=tuple(str(value) for value in strata_data["balancing"]),
        ),
        base_model=BaseModelConfig(
            version=str(model_data["version"]),
            data_source=str(model_data["data_source"]),
            upstream_commit=str(model_data["upstream_commit"]),
            checkpoint_path=str(model_data["checkpoint_path"]),
            checkpoint_sha256=str(model_data["checkpoint_sha256"]),
            s2_bands=tuple(model_data.get("s2_bands", CANONICAL_S2_BANDS)),
        ),
        temporal_windows=TemporalWindowsConfig(
            boundary=str(window_data["boundary"]),
            minimum_days=int(window_data["minimum_days"]),
            maximum_days=int(window_data["maximum_days"]),
            anchor_days=tuple(int(value) for value in window_data["anchor_days"]),
            anchor_probability=float(window_data["anchor_probability"]),
            prefix_probability=float(window_data["prefix_probability"]),
            view_dropout_probability=float(window_data["view_dropout_probability"]),
            evaluation_days=tuple(
                int(value) for value in window_data["evaluation_days"]
            ),
            cross_year_policy=str(window_data["cross_year_policy"]),
            teacher_target=str(window_data["teacher_target"]),
            teacher_anchor_full_weight_days=int(
                window_data["teacher_anchor_full_weight_days"]
            ),
            teacher_anchor_full_observations=int(
                window_data["teacher_anchor_full_observations"]
            ),
        ),
        student=StudentConfig(
            model_dim=int(student_data["model_dim"]),
            layers=int(student_data["layers"]),
            heads=int(student_data["heads"]),
            feedforward_dim=int(student_data["feedforward_dim"]),
            output_dim=int(student_data["output_dim"]),
            temporal_view_sizes=tuple(int(value) for value in student_data["temporal_view_sizes"]),
        ),
        lora=LoRAExperimentConfig(
            ranks=tuple(int(value) for value in lora_data["ranks"]),
            targets=tuple(str(value) for value in lora_data["targets"]),
            base_anchor_weights=tuple(float(value) for value in lora_data["base_anchor_weights"]),
            window_conditioner=bool(lora_data["window_conditioner"]),
        ),
        raw=data,
    )
    config.validate(
        require_checkpoint=require_checkpoint, require_boundaries=require_boundaries
    )
    return config
