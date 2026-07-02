from pathlib import Path

import pytest
import yaml

from spectrajam.config import load_config
from spectrajam.contracts import CANONICAL_S2_BANDS, ContractError, require_band_order
from spectrajam.normalization import get_stats

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["smoke.yaml", "pilot.yaml", "preferred-full.yaml"])
def test_committed_configs_are_valid(name: str) -> None:
    config = load_config(ROOT / "configs" / name)
    assert config.base_model.s2_bands == CANONICAL_S2_BANDS
    assert config.countries == ("RWA", "ISR")
    assert {7, 14} <= set(config.temporal_windows.anchor_days)
    assert {7, 14} <= set(config.temporal_windows.evaluation_days)
    assert config.temporal_windows.teacher_target == "same_window"
    assert config.lora.window_conditioner


def test_band_order_is_a_hard_contract() -> None:
    with pytest.raises(ContractError, match="checkpoint-critical"):
        require_band_order(("B02", "B03", "B04", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"))


def test_normalization_is_source_specific() -> None:
    assert get_stats("mpc") != get_stats("aws")
    assert len(get_stats("mpc").s2_mean) == 10


def test_stac_and_checkpoint_source_cannot_be_mixed(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "configs" / "pilot.yaml").read_text())
    data["base_model"]["data_source"] = "aws"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ContractError, match="must use the MPC checkpoint"):
        load_config(path)


def test_operational_validation_rejects_missing_boundary(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "configs" / "pilot.yaml").read_text())
    for policy in data["extent_policy"].values():
        policy["boundary_path"] = str(tmp_path / "missing.gpkg")
    path = tmp_path / "missing-boundary.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ContractError, match="boundary not found"):
        load_config(
            path,
            require_boundaries=True,
        )


def test_annual_teacher_target_is_rejected_for_short_windows(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "configs" / "pilot.yaml").read_text())
    data["temporal_windows"]["teacher_target"] = "annual"
    path = tmp_path / "leaky.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ContractError, match="same_window"):
        load_config(path)


def test_unknown_config_keys_are_rejected_instead_of_ignored(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "configs" / "pilot.yaml").read_text())
    data["lora"]["imaginary_option"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ContractError, match="unexpected"):
        load_config(path)


def test_checkpoint_digest_must_be_pinned(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    data["base_model"]["checkpoint_sha256"] = "REPLACE_BEFORE_TRAINING"
    path = tmp_path / "unpinned.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ContractError, match="checkpoint_sha256"):
        load_config(path)
