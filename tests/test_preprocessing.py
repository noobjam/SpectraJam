from datetime import date

import numpy as np

from spectrajam.preprocessing import (
    amplitude_to_mpc_s1,
    harmonize_mpc_s2,
    valid_scl,
)


def test_scl_contract_matches_pinned_tessera() -> None:
    classes = np.arange(12, dtype=np.uint8)
    assert valid_scl(classes).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


def test_s2_baseline_harmonization_is_source_and_date_specific() -> None:
    values = np.array([0, 999, 1000, 2500, np.nan], dtype=np.float32)
    assert harmonize_mpc_s2(values, date(2022, 1, 25)).tolist() == [
        0,
        999,
        1000,
        2500,
        0,
    ]
    assert harmonize_mpc_s2(values, date(2022, 1, 26)).tolist() == [
        0,
        999,
        0,
        1500,
        0,
    ]


def test_s1_amplitude_conversion_matches_upstream_truncation() -> None:
    values = np.array([0.0, np.nan, 0.01, 0.1, 1.0, 10.0], dtype=np.float32)
    converted = amplitude_to_mpc_s1(values)
    assert converted.dtype == np.int16
    assert converted.tolist() == [0, 0, 2000, 6000, 10000, 14000]
