from collections import Counter

import pytest

from spectrajam.contracts import ContractError
from spectrajam.sampling import (
    Candidate,
    allocate_strata,
    assert_no_block_leakage,
    expand_years,
    select_candidates,
    validate_manifest_universe,
)


def test_sqrt_allocation_conserves_total_and_tempers_majority() -> None:
    allocation = allocate_strata(
        {"common": 900, "rare": 100},
        total=200,
        min_per_stratum=0,
        allocation_power=0.5,
    )
    assert sum(allocation.values()) == 200
    assert allocation["rare"] > 20
    assert allocation["rare"] <= 100


def test_floor_cannot_exceed_budget() -> None:
    with pytest.raises(ContractError, match="reserves more"):
        allocate_strata({"a": 10, "b": 10}, 5, min_per_stratum=3, allocation_power=0.5)


def _candidates() -> list[Candidate]:
    values = []
    for country in ("RWA", "ISR"):
        for index in range(20):
            values.append(
                Candidate(
                    candidate_id=f"{country}-{index}",
                    country=country,
                    longitude=30.0 + index / 100,
                    latitude=-2.0 + index / 100,
                    spatial_block=f"shared-block-{index // 2}",
                    stratum="forest" if index < 15 else "water",
                )
            )
    return values


def test_selection_is_deterministic_and_block_safe() -> None:
    kwargs = dict(
        points_per_country=10,
        min_per_stratum=2,
        allocation_power=0.5,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=7,
    )
    first = select_candidates(_candidates(), **kwargs)
    second = select_candidates(_candidates(), **kwargs)
    assert first == second
    assert Counter(point.candidate.country for point in first) == {"ISR": 10, "RWA": 10}
    assert_no_block_leakage(first)
    assert all(0 < point.inclusion_probability <= 1 for point in first)


def test_duplicate_candidate_ids_are_rejected() -> None:
    duplicate = [_candidates()[0], _candidates()[0]]
    with pytest.raises(ContractError, match="duplicate"):
        select_candidates(
            duplicate,
            points_per_country=1,
            min_per_stratum=0,
            allocation_power=0.5,
            split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=1,
        )


def test_one_country_candidate_universe_is_rejected() -> None:
    with pytest.raises(ContractError, match="candidate countries"):
        select_candidates(
            [candidate for candidate in _candidates() if candidate.country == "RWA"],
            points_per_country=5,
            min_per_stratum=0,
            allocation_power=0.5,
            split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=1,
        )


def test_manifest_expands_independent_spatial_and_year_splits() -> None:
    point = select_candidates(
        _candidates(),
        points_per_country=1,
        min_per_stratum=0,
        allocation_power=0.5,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=3,
    )[0]
    rows = list(expand_years(
        [point],
        {"train": [2019, 2020], "validation": [2024], "test": [2025]},
    ))
    assert len(rows) == 4
    assert {row.year_split for row in rows} == {"train", "validation", "test"}
    assert {row.spatial_split for row in rows} == {point.split}


def test_manifest_validator_rejects_missing_years() -> None:
    points = select_candidates(
        _candidates(),
        points_per_country=1,
        min_per_stratum=0,
        allocation_power=0.5,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=3,
    )
    rows = list(
        expand_years(points, {"train": [2019], "validation": [2024], "test": [2025]})
    )
    with pytest.raises(ContractError, match="has years"):
        list(
            validate_manifest_universe(
                rows[:-1],
                ["RWA", "ISR"],
                {"train": [2019], "validation": [2024], "test": [2025]},
                expected_points_per_country=1,
            )
        )
