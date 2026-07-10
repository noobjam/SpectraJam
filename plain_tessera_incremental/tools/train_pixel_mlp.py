from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

from spectrajam.contracts import sha256_file


def _require_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:  # pragma: no cover - exercised on the GPU VM
        raise RuntimeError(
            "numpy, pandas, pyarrow, and a CUDA-compatible PyTorch build are required"
        ) from error
    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
    }


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    class_names: list[str],
    np: Any,
) -> tuple[dict[str, Any], Any]:
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    rows = []
    f1_values = []
    recalls = []
    weighted_f1 = 0.0
    total = int(matrix.sum())
    for class_id, label in enumerate(class_names):
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        false_negative = int(matrix[class_id, :].sum() - true_positive)
        support = int(matrix[class_id, :].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        weighted_f1 += f1 * support
        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metrics = {
        "n": total,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": float(weighted_f1 / total) if total else 0.0,
        "per_class": rows,
    }
    return metrics, matrix


def _make_model(input_dim: int, hidden_dims: list[int], class_count: int, dropout: float, nn: Any):
    if not hidden_dims or any(value < 1 for value in hidden_dims):
        raise ValueError("hidden dimensions must be positive")
    layers: list[Any] = [nn.LayerNorm(input_dim)]
    width = input_dim
    for hidden in hidden_dims:
        layers.extend(
            [
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        width = hidden
    layers.append(nn.Linear(width, class_count))
    return nn.Sequential(*layers)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def _write_csv_atomic(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_csv(part, index=False)
    os.replace(part, path)


def _write_parquet_atomic(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False, compression="zstd")
    os.replace(part, path)


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    class_names: list[str],
    dependencies: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any, Any]:
    np = dependencies["np"]
    torch = dependencies["torch"]
    model.eval()
    losses = []
    truth = []
    probabilities = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(features)
            losses.append(float(criterion(logits, targets).item()) * len(targets))
            truth.append(targets.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(truth)
    probability = np.concatenate(probabilities)
    y_pred = probability.argmax(axis=1)
    metrics, matrix = classification_metrics(y_true, y_pred, class_names, np)
    metrics["loss"] = sum(losses) / len(y_true)
    return metrics, matrix, y_true, probability


def train(args: argparse.Namespace) -> dict[str, Any]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    pd = dependencies["pd"]
    torch = dependencies["torch"]
    DataLoader = dependencies["DataLoader"]
    TensorDataset = dependencies["TensorDataset"]

    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"classification dataset not found: {dataset_path}")
    manifest_path = (
        Path(args.dataset_manifest).expanduser()
        if args.dataset_manifest
        else dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("output_sha256") != sha256_file(dataset_path):
        raise RuntimeError("classification dataset SHA-256 does not match its manifest")

    frame = pd.read_parquet(dataset_path)
    required = {"pixel_id", "label", "class_id", "split", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"classification dataset is missing columns: {sorted(missing)}")
    class_rows = frame[["label", "class_id"]].drop_duplicates()
    if class_rows["label"].duplicated().any() or class_rows["class_id"].duplicated().any():
        raise RuntimeError("label/class_id mapping is not one-to-one")
    class_rows = class_rows.sort_values("class_id")
    expected_ids = list(range(len(class_rows)))
    if class_rows["class_id"].astype(int).tolist() != expected_ids:
        raise RuntimeError("class IDs must be contiguous from zero")
    class_names = class_rows["label"].astype(str).tolist()
    features = np.vstack(
        frame["embedding"].map(lambda value: np.asarray(value, dtype=np.float32))
    )
    if features.shape != (len(frame), 128) or not np.isfinite(features).all():
        raise RuntimeError(f"expected finite [N,128] embeddings, got {features.shape}")
    targets = frame["class_id"].to_numpy(np.int64)
    split_indices = {
        name: np.flatnonzero(frame["split"].astype(str).eq(name).to_numpy())
        for name in ("train", "validation", "test")
    }
    if any(not len(indices) for indices in split_indices.values()):
        raise RuntimeError("train, validation, and test splits must all be non-empty")
    train_indices = split_indices["train"]
    mean = features[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    features = (features - mean) / scale

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device, torch)
    model = _make_model(
        128,
        args.hidden_dims,
        len(class_names),
        args.dropout,
        dependencies["nn"],
    ).to(device)

    class_counts = np.bincount(targets[train_indices], minlength=len(class_names))
    if (class_counts == 0).any():
        raise RuntimeError("every class must be represented in the training split")
    class_weights = np.sqrt(class_counts.sum() / (len(class_names) * class_counts))
    class_weights = class_weights / class_weights.mean()
    criterion = dependencies["nn"].CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    tensors = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            torch.utils.data.Subset(tensors, indices.tolist()),
            batch_size=args.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, indices in split_indices.items()
    }

    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_rows = 0
        for batch_features, batch_targets in loaders["train"]:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_targets)
            train_rows += len(batch_targets)
        validation, _, _, _ = _evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            class_names,
            dependencies,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_rows,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation["macro_f1"] > best_macro_f1 + 1e-8:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation, validation_matrix, _, _ = _evaluate(
        model,
        loaders["validation"],
        criterion,
        device,
        class_names,
        dependencies,
    )
    test, test_matrix, test_truth, test_probability = _evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        class_names,
        dependencies,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    checkpoint_part = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
    torch.save(
        {
            "schema": "pixel-landcover-mlp-v1",
            "state_dict": best_state,
            "input_dim": 128,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "class_names": class_names,
            "normalization_mean": torch.as_tensor(mean),
            "normalization_scale": torch.as_tensor(scale),
            "dataset_sha256": sha256_file(dataset_path),
            "best_epoch": best_epoch,
        },
        checkpoint_part,
    )
    os.replace(checkpoint_part, checkpoint_path)
    history_frame = pd.DataFrame(history)
    _write_csv_atomic(history_frame, output_dir / "training_history.csv")
    confusion_rows = []
    for split, matrix in (
        ("validation", validation_matrix),
        ("test", test_matrix),
    ):
        for true_id, true_label in enumerate(class_names):
            for predicted_id, predicted_label in enumerate(class_names):
                confusion_rows.append(
                    {
                        "split": split,
                        "true_class_id": true_id,
                        "true_label": true_label,
                        "predicted_class_id": predicted_id,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_id, predicted_id]),
                    }
                )
    _write_csv_atomic(pd.DataFrame(confusion_rows), output_dir / "confusion_matrix.csv")
    test_frame = frame.iloc[split_indices["test"]][
        ["pixel_id", "label", "class_id", "spatial_block"]
    ].copy()
    test_prediction = test_probability.argmax(axis=1)
    if not np.array_equal(test_frame["class_id"].to_numpy(np.int64), test_truth):
        raise RuntimeError("test prediction order changed during evaluation")
    test_frame["predicted_class_id"] = test_prediction
    test_frame["predicted_label"] = [class_names[value] for value in test_prediction]
    test_frame["probabilities"] = [row.astype(np.float32) for row in test_probability]
    _write_parquet_atomic(test_frame, output_dir / "test_predictions.parquet")
    metrics = {
        "schema": "pixel-landcover-mlp-evaluation-v1",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest": str(manifest_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "model": {
            "input_dim": 128,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "optimization": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "class_weights": {
                label: float(class_weights[index])
                for index, label in enumerate(class_names)
            },
        },
        "validation": validation,
        "test": test,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "history": str(output_dir / "training_history.csv"),
            "confusion_matrix": str(output_dir / "confusion_matrix.csv"),
            "test_predictions": str(output_dir / "test_predictions.parquet"),
        },
    }
    _write_json_atomic(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a spatially held-out MLP on 128-dimensional pixel embeddings"
    )
    parser.add_argument(
        "--dataset",
        default="/mnt/noobjam/rwanda_worldcover_mlp/pixel_classification_w2.parquet",
    )
    parser.add_argument("--dataset-manifest")
    parser.add_argument(
        "--output-dir",
        default="/mnt/noobjam/rwanda_worldcover_mlp/mlp_w2",
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=24_051_995)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
