from __future__ import annotations

from pathlib import Path

import pytest

from spectrajam.contracts import ContractError
from spectrajam.frame_sources import (
    FRAME_SOURCES,
    fetch_frame_sources,
    select_frame_sources,
    verify_frame_sources,
    write_frame_sources_receipt,
)

EXPECTED_KEYS = (
    "world_bank_admin0",
    "world_bank_ndlsa",
    "world_bank_data_dictionary",
    "resolve_ecoregions_2017",
    "worldcover_2021_grid",
    "worldcover_2021_s03e027",
    "worldcover_2021_s03e030",
    "worldcover_2021_n27e033",
    "worldcover_2021_n30e033",
    "worldcover_2021_n33e033",
)


def test_registry_is_complete_pinned_and_has_unique_destinations() -> None:
    assert tuple(FRAME_SOURCES) == EXPECTED_KEYS
    assert tuple(source.key for source in select_frame_sources()) == EXPECTED_KEYS
    assert len({source.relative_destination for source in FRAME_SOURCES.values()}) == 10
    assert sum(
        source.artifact.expected_bytes for source in FRAME_SOURCES.values()
    ) == 506_016_614
    assert all(source.artifact.url.startswith("https://") for source in FRAME_SOURCES.values())
    assert all(len(source.artifact.sha256) == 64 for source in FRAME_SOURCES.values())

    admin0 = FRAME_SOURCES["world_bank_admin0"]
    assert admin0.artifact.expected_bytes == 63_422_464
    assert admin0.artifact.sha256 == (
        "97f0c8a0fa848b9a8414dbeb2e058fa37d59b13794ec232a87da000bdf4b117e"
    )
    assert dict(admin0.metadata)["layer"] == "WB_GAD_ADM0"
    assert "versionid=2026-06-12" in admin0.artifact.url
    assert dict(admin0.metadata)["dataset_terms"].startswith("https://www.worldbank.org/")

    ndlsa = FRAME_SOURCES["world_bank_ndlsa"]
    assert ndlsa.artifact.expected_bytes == 684_032
    assert dict(ndlsa.metadata)["layer"] == "WB_GAD_ADM0_NDLSA"
    assert dict(ndlsa.metadata)["policy"] == "exclude"

    resolve = FRAME_SOURCES["resolve_ecoregions_2017"]
    assert resolve.artifact.expected_bytes == 149_248_653
    assert dict(resolve.metadata)["primary_field"] == "ECO_ID"
    assert dict(resolve.metadata)["dbf_encoding"] == "ISO-8859-1"


def test_worldcover_registry_has_the_exact_country_tiles() -> None:
    tiles = [
        source
        for source in FRAME_SOURCES.values()
        if source.role == "land-cover-map"
    ]
    assert [dict(source.metadata)["tile"] for source in tiles] == [
        "S03E027",
        "S03E030",
        "N27E033",
        "N30E033",
        "N33E033",
    ]
    assert sum(source.artifact.expected_bytes for source in tiles) == 292_101_324
    assert all(source.version == "v200 (2.0.0)" for source in tiles)
    assert all(dict(source.metadata)["nodata"] == 0 for source in tiles)


def test_selection_is_deduplicated_and_returned_in_registry_order() -> None:
    selected = select_frame_sources(
        [
            "worldcover_2021_n33e033",
            "world_bank_admin0",
            "worldcover_2021_n33e033",
        ]
    )
    assert [source.key for source in selected] == [
        "world_bank_admin0",
        "worldcover_2021_n33e033",
    ]
    assert select_frame_sources("world_bank_ndlsa")[0].key == "world_bank_ndlsa"

    with pytest.raises(ContractError, match="at least one"):
        select_frame_sources([])
    with pytest.raises(ContractError, match="unknown"):
        select_frame_sources(["not_a_source"])


def test_fetch_helper_uses_deterministic_paths_and_returns_provenance(tmp_path: Path) -> None:
    calls = []

    def fake_fetch(artifact, destination, **options):
        destination = Path(destination)
        calls.append((artifact, destination, options))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch(exist_ok=True)
        return destination

    existing = FRAME_SOURCES["world_bank_admin0"].destination(tmp_path)
    existing.parent.mkdir(parents=True)
    existing.touch()
    results = fetch_frame_sources(
        tmp_path,
        ["worldcover_2021_s03e027", "world_bank_admin0"],
        fetcher=fake_fetch,
        verifier=lambda path, artifact: None,
        max_attempts=3,
    )

    assert [result.source.key for result in results] == [
        "world_bank_admin0",
        "worldcover_2021_s03e027",
    ]
    assert [result.reused for result in results] == [True, False]
    assert calls[0][1] == tmp_path / "boundaries/world-bank-v2/admin0.gpkg"
    assert calls[1][1] == tmp_path / (
        "worldcover/2021-v200/map/"
        "ESA_WorldCover_10m_2021_v200_S03E027_Map.tif"
    )
    assert all(call[2] == {"max_attempts": 3} for call in calls)

    receipt = results[1].provenance()
    assert receipt["key"] == "worldcover_2021_s03e027"
    assert receipt["expected_bytes"] == 77_269_683
    assert receipt["verified"] is True
    assert receipt["reused"] is False
    assert receipt["metadata"]["tile"] == "S03E027"


def test_provenance_contains_license_product_and_pin(tmp_path: Path) -> None:
    source = FRAME_SOURCES["world_bank_data_dictionary"]
    receipt = source.provenance(source.destination(tmp_path))
    assert receipt["producer"] == "World Bank"
    assert receipt["product"] == "World Bank Official Boundaries Data Dictionary"
    assert receipt["version"] == "2"
    assert receipt["license"] == "CC-BY-4.0"
    assert receipt["sha256"] == source.artifact.sha256
    assert receipt["checksum_provenance"] == "independently-observed-2026-07-02"
    assert receipt["metadata"]["release_id"] == "DR0095372"


def test_source_receipt_is_deterministic_and_refuses_different_content(
    tmp_path: Path,
) -> None:
    source = FRAME_SOURCES["world_bank_admin0"]
    fetched = fetch_frame_sources(
        tmp_path,
        source.key,
        fetcher=lambda artifact, destination: Path(destination),
        verifier=lambda path, artifact: None,
    )
    receipt = tmp_path / "sources.lock.json"
    assert write_frame_sources_receipt(receipt, fetched) == receipt
    first = receipt.read_bytes()
    assert write_frame_sources_receipt(receipt, fetched) == receipt
    assert receipt.read_bytes() == first

    receipt.write_text("different")
    with pytest.raises(ContractError, match="refusing to replace"):
        write_frame_sources_receipt(receipt, fetched)


def test_verify_sources_fails_closed_before_reading_data(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not found"):
        verify_frame_sources(tmp_path)

    source = FRAME_SOURCES["world_bank_admin0"]
    path = source.destination(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    with pytest.raises(ContractError, match="byte count"):
        verify_frame_sources(tmp_path)


def test_fetch_helper_does_not_trust_an_injected_fetcher(tmp_path: Path) -> None:
    def corrupt_fetch(artifact, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"wrong")
        return destination

    with pytest.raises(ContractError, match="byte count"):
        fetch_frame_sources(
            tmp_path,
            "world_bank_admin0",
            fetcher=corrupt_fetch,
        )
