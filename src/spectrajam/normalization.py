from __future__ import annotations

from dataclasses import dataclass

from .contracts import CANONICAL_S2_BANDS, ContractError


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    s2_mean: tuple[float, ...]
    s2_std: tuple[float, ...]
    s1_ascending_mean: tuple[float, ...]
    s1_ascending_std: tuple[float, ...]
    s1_descending_mean: tuple[float, ...]
    s1_descending_std: tuple[float, ...]


# Copied from official TESSERA v1.1 commit d06ee44... (MIT), keyed by the
# preprocessing source. These numbers and the checkpoint must never be mixed.
STATS = {
    "mpc": NormalizationStats(
        s2_mean=(2683.4553, 2223.3630, 2432.0950, 3633.1970, 3602.1755,
                 3006.4324, 3400.2710, 3515.6392, 2456.9163, 1983.8783),
        s2_std=(2739.5217, 2846.2993, 2690.8250, 2290.0439, 2088.8970,
                2673.1106, 2381.4521, 2229.5225, 1601.0942, 1495.3545),
        s1_ascending_mean=(5588.3291, 3025.6270),
        s1_ascending_std=(1713.4646, 1693.0471),
        s1_descending_mean=(5552.9683, 2955.0520),
        s1_descending_std=(1685.5857, 1677.6414),
    ),
    "aws": NormalizationStats(
        s2_mean=(2793.6589, 2356.7776, 2551.0496, 3741.9229, 3713.7844,
                 3120.1997, 3516.3342, 3637.0342, 2501.0283, 2038.1504),
        s2_std=(2810.0093, 2933.8835, 2755.6360, 2344.5027, 2145.7986,
                2743.9019, 2438.8601, 2286.5977, 1680.7367, 1585.5529),
        s1_ascending_mean=(5697.0859, 2838.6687),
        s1_ascending_std=(1671.3737, 1789.4116),
        s1_descending_mean=(5759.1367, 2873.2854),
        s1_descending_std=(1583.2858, 1747.8390),
    ),
}


def get_stats(data_source: str) -> NormalizationStats:
    try:
        stats = STATS[data_source.lower()]
    except KeyError as error:
        raise ContractError(f"unknown TESSERA v1.1 data source: {data_source}") from error
    if len(stats.s2_mean) != len(CANONICAL_S2_BANDS):
        raise AssertionError("normalization table does not match the canonical band contract")
    return stats
