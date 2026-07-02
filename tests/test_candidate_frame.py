from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spectrajam.candidate_frame import (
    _assign_ecoregions,
    _load_boundaries,
    _require_optional_dependencies,
    candidate_frame_contract,
    candidate_id,
    lattice_center,
    lattice_index_bounds,
    spatial_block_id,
    verify_candidate_frame_receipt,
    worldcover_tile_id,
    write_json_atomic,
)
from spectrajam.config import load_config
from spectrajam.contracts import ContractError, sha256_file

ROOT = Path(__file__).parents[1]


def test_lattice_uses_globally_anchored_cell_centers() -> None:
    assert lattice_index_bounds(0, 999, 200) == (0, 4)
    assert lattice_index_bounds(-101, 101, 200) == (-1, 0)
    assert [lattice_center(index, 200) for index in range(-1, 2)] == [
        -100.0,
        100.0,
        300.0,
    ]


def test_lattice_rejects_invalid_bounds_and_spacing() -> None:
    with pytest.raises(ContractError, match="bounds"):
        lattice_index_bounds(2, 1, 200)
    with pytest.raises(ContractError, match="spacing"):
        lattice_index_bounds(0, 1, 0)
    with pytest.raises(ContractError, match="no lattice"):
        lattice_index_bounds(0, 10, 200)


@pytest.mark.parametrize(
    ("longitude", "latitude", "tile"),
    [
        (28.9, -2.0, "S03E027"),
        (30.0, -2.0, "S03E030"),
        (34.8, 29.999999, "N27E033"),
        (34.8, 30.0, "N27E033"),
        (34.8, 30.000001, "N30E033"),
        (35.0, 33.0, "N30E033"),
        (35.0, 33.000001, "N33E033"),
        (-0.1, -0.1, "S03W003"),
    ],
)
def test_worldcover_tile_ids_use_half_open_three_degree_cells(
    longitude: float, latitude: float, tile: str
) -> None:
    assert worldcover_tile_id(longitude, latitude) == tile


def test_worldcover_tile_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ContractError, match="invalid WGS84"):
        worldcover_tile_id(180, 0)
    with pytest.raises(ContractError, match="invalid WGS84"):
        worldcover_tile_id(0, 90)
    with pytest.raises(ContractError, match="invalid WGS84"):
        worldcover_tile_id(0, -90)


def test_candidate_and_block_ids_pin_projection_and_grid_indices() -> None:
    assert candidate_id("RWA", 32735, 100, 200) == "RWA-e32735-x100-y200"
    assert spatial_block_id("ISR", 32636, 19_999.9, 20_000, 20_000) == (
        "ISR-e32636-bx0-by1"
    )


def test_candidate_receipt_binds_exact_csv_bytes(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("candidate_id,country\na,RWA\n")
    receipt = tmp_path / "candidates.receipt.json"
    write_json_atomic(
        receipt,
        {
            "schema": "spectrajam-candidate-frame-v1",
            "candidate_output": {
                "bytes": candidates.stat().st_size,
                "sha256": sha256_file(candidates),
            },
        },
    )
    assert verify_candidate_frame_receipt(candidates, receipt)["schema"].endswith("v1")
    candidates.write_text("candidate_id,country\nb,ISR\n")
    with pytest.raises(ContractError, match="SHA-256"):
        verify_candidate_frame_receipt(candidates, receipt)


def test_frame_contract_ignores_unrelated_training_tier_settings() -> None:
    smoke = load_config(ROOT / "configs/smoke.yaml")
    pilot = load_config(ROOT / "configs/pilot.yaml")
    full = load_config(ROOT / "configs/preferred-full.yaml")
    assert candidate_frame_contract(smoke) == candidate_frame_contract(pilot)
    assert candidate_frame_contract(smoke) == candidate_frame_contract(full)


def test_ecoregion_assignment_preserves_gaps_and_resolves_ties_to_lowest_id() -> None:
    geopandas = pytest.importorskip("geopandas")
    numpy = pytest.importorskip("numpy")
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import box
    from shapely.strtree import STRtree

    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    frame = geopandas.GeoDataFrame(
        {"ECO_ID": [20, 10]}, geometry=geometries, crs="EPSG:4326"
    )
    assignments, ambiguous = _assign_ecoregions(
        numpy.array([0.5, 1.0, 3.0]),
        numpy.array([0.5, 0.5, 0.5]),
        frame,
        STRtree(frame.geometry.to_numpy()),
        {"np": numpy, "shapely": shapely},
    )
    assert assignments.tolist() == [20, 10, -1]
    assert ambiguous == 1


def _write_boundary_fixture(tmp_path, *, overlap: bool):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    admin0_path = tmp_path / "admin0.gpkg"
    ndlsa_path = tmp_path / "ndlsa.gpkg"
    admin0 = geopandas.GeoDataFrame(
        {
            "ISO_A3": ["RWA", "ISR"],
            "WB_STATUS": ["Member State", "Member State"],
            "NAM_0": ["Rwanda", "Israel"],
        },
        geometry=[box(29, -3, 31, -1), box(34, 29, 36, 34)],
        crs="EPSG:4326",
    )
    admin0.to_file(admin0_path, layer="WB_GAD_ADM0", driver="GPKG", engine="pyogrio")

    geometries = [box(40 + index, 40, 40.5 + index, 40.5) for index in range(24)]
    geometries[0] = box(35, 30, 35.1, 30.1) if overlap else box(36, 30, 36.1, 30.1)
    ndlsa = geopandas.GeoDataFrame(
        {"WB_STATUS": ["Non-determined legal status area"] * 24},
        geometry=geometries,
        crs="EPSG:4326",
    )
    ndlsa.to_file(
        ndlsa_path,
        layer="WB_GAD_ADM0_NDLSA",
        driver="GPKG",
        engine="pyogrio",
    )
    config = SimpleNamespace(
        countries=("RWA", "ISR"),
        extents={
            "RWA": SimpleNamespace(boundary_path=str(admin0_path)),
            "ISR": SimpleNamespace(boundary_path=str(admin0_path)),
        },
    )
    return config, ndlsa_path


def test_boundary_loader_selects_exact_country_rows_and_ndlsa_touches(tmp_path) -> None:
    config, ndlsa_path = _write_boundary_fixture(tmp_path, overlap=False)
    boundaries, touches = _load_boundaries(
        config,
        config.extents["RWA"].boundary_path,
        ndlsa_path,
        _require_optional_dependencies(),
    )
    assert set(boundaries) == {"RWA", "ISR"}
    assert touches == {"RWA": 0, "ISR": 1}


def test_boundary_loader_rejects_positive_area_ndlsa_overlap(tmp_path) -> None:
    config, ndlsa_path = _write_boundary_fixture(tmp_path, overlap=True)
    with pytest.raises(ContractError, match="overlaps"):
        _load_boundaries(
            config,
            config.extents["RWA"].boundary_path,
            ndlsa_path,
            _require_optional_dependencies(),
        )
