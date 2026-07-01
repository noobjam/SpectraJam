from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .contracts import ContractError, sha256_file
from .models.tessera_v11 import load_tessera_v11


def verify_parity_fixture(
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    fixture_path: str | Path,
    receipt_path: str | Path,
    absolute_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Compare SpectraJam with an output captured from pinned upstream inference."""
    fixture = np.load(fixture_path)
    required = {"s2", "s1", "expected"}
    if not required.issubset(fixture.files):
        raise ContractError(f"parity fixture must contain {sorted(required)}")
    model = load_tessera_v11(checkpoint_path, checkpoint_sha256, device="cpu")
    model.eval()
    with torch.inference_mode():
        actual = model(
            torch.from_numpy(fixture["s2"]).float(),
            torch.from_numpy(fixture["s1"]).float(),
        ).cpu().numpy()
    expected = fixture["expected"].astype(np.float32, copy=False)
    if actual.shape != expected.shape:
        raise ContractError(f"parity shape mismatch: {actual.shape} != {expected.shape}")
    maximum_error = float(np.max(np.abs(actual - expected)))
    if maximum_error > absolute_tolerance:
        raise ContractError(
            f"upstream parity failed: max_abs_error={maximum_error} > {absolute_tolerance}"
        )
    receipt = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "fixture_sha256": sha256_file(fixture_path),
        "maximum_absolute_error": maximum_error,
        "absolute_tolerance": absolute_tolerance,
        "passed": True,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True).encode()
    destination = Path(receipt_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    return receipt


def parity_receipt_matches(receipt_path: str | Path, checkpoint_sha256: str) -> bool:
    try:
        receipt = json.loads(Path(receipt_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return bool(receipt.get("passed")) and receipt.get("checkpoint_sha256") == checkpoint_sha256
