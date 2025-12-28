"""
Shared utilities for the 3D CNN experiments.

This module centralizes dataset loading, model creation, training and
evaluation so that the three experiment entrypoints remain concise
and consistent. It sticks closely to the behavior described in the
original README for the 2025 baseline: 5-fold cross-validation with
simple oversampling, R3D-18 architecture, and ROC/AUC computed from
probability outputs.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models


DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_DIR", "C:/dataset/runs"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "C:/dataset/data"))


@dataclass
class ExperimentConfig:
    name: str
    pretrained: bool
    batch_size: int = 2
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    holdout_test_size: float = 0.15

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class FoldMetrics:
    fold: int
    accuracy: float
    auc: float
    f1: float
    loss: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class DependencyVersions:
    torch: str
    torchvision: str
    numpy: str
    sklearn: str

    @classmethod
    def capture(cls) -> "DependencyVersions":
        import sklearn

        return cls(
            torch=torch.__version__,
            torchvision=models.__version__,
            numpy=np.__version__,
            sklearn=sklearn.__version__,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class Knee3DDataset(Dataset):
    """Minimal Dataset wrapper for preprocessed 3D DESS volumes."""

    def __init__(self, volumes: np.ndarray, labels: np.ndarray, indices: Iterable[int]):
        self.volumes = volumes.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.indices = np.array(list(indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        volume = self.volumes[self.indices[idx]]  # (H, W, D)
        # Reorder to (C=1, T=D, H, W) expected by R3D-18
        tensor = torch.from_numpy(volume).permute(2, 0, 1).unsqueeze(0)
        return tensor, int(self.labels[self.indices[idx]])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_numpy_volumes(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    normal_path = data_dir / "normal-3DESS-128-64.npy"
    abnormal_path = data_dir / "abnormal-3DESS-128-64.npy"
    if not normal_path.exists() or not abnormal_path.exists():
        raise FileNotFoundError(
            f"Expected dataset files at {normal_path} and {abnormal_path}. Set DATA_DIR if they live elsewhere."
        )

    normal = np.load(normal_path)
    abnormal = np.load(abnormal_path)
    features = np.concatenate([normal, abnormal], axis=0)
    labels = np.concatenate(
        [np.zeros(len(normal), dtype=np.int64), np.ones(len(abnormal), dtype=np.int64)], axis=0
    )
    return features, labels


def build_model(pretrained: bool, device: torch.device) -> nn.Module:
    model = models.video.r3d_18(pretrained=pretrained)
    # Binary classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model.to(device)


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_sample_counts = np.bincount(labels)
    # Avoid division by zero if a class is absent (unlikely after concatenation)
    weights = 1.0 / np.maximum(class_sample_counts, 1)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray, List[float]]:
    model.eval()
    losses: List[float] = []
    all_targets: List[int] = []
    all_probs: List[float] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.float().to(device)
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)
        losses.append(loss.item() * inputs.size(0))
        all_targets.extend(targets.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    mean_loss = float(np.sum(losses) / len(loader.dataset))
    targets_arr = np.array(all_targets)
    probs_arr = np.array(all_probs)
    return mean_loss, targets_arr, probs_arr, all_probs


def compute_metrics(targets: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "auc": float(roc_auc_score(targets, probs)),
        "f1": float(f1_score(targets, preds)),
    }


def summarize_folds(fold_metrics: List[FoldMetrics]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in ["accuracy", "auc", "f1", "loss"]:
        values = np.array([getattr(m, key) for m in fold_metrics], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = float(1.96 * std / math.sqrt(len(values))) if len(values) > 0 else 0.0
        summary[key] = {"mean": mean, "std": std, "ci95": ci95}
    return summary


def save_fold_outputs(
    run_dir: Path,
    fold_idx: int,
    targets: np.ndarray,
    probs: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    metrics_path = run_dir / f"fold{fold_idx}_predictions.csv"
    roc_path = run_dir / f"fold{fold_idx}_roc.csv"

    np.savetxt(metrics_path, np.column_stack((targets, probs)), delimiter=",", header="target,prob", comments="")
    np.savetxt(roc_path, np.column_stack((fpr, tpr, thresholds)), delimiter=",", header="fpr,tpr,threshold", comments="")


def write_metadata(run_dir: Path, config: ExperimentConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
    versions = DependencyVersions.capture()
    (run_dir / "versions.json").write_text(versions.to_json(), encoding="utf-8")


def cross_validation_experiment(config: ExperimentConfig) -> Dict[str, Dict[str, float]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volumes, labels = load_numpy_volumes(DATA_DIR)

    run_dir = DEFAULT_RUNS_ROOT / config.name
    write_metadata(run_dir, config)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    fold_metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(volumes, labels)):
        train_dataset = Knee3DDataset(volumes, labels, train_idx)
        val_dataset = Knee3DDataset(volumes, labels, val_idx)

        sampler = make_sampler(labels[train_idx])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = build_model(pretrained=config.pretrained, device=device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = -1.0
        best_state: Dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        best_targets: np.ndarray | None = None
        best_probs: np.ndarray | None = None

        for _ in range(config.num_epochs):
            run_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, targets, probs, _ = evaluate(model, val_loader, criterion, device)
            metrics = compute_metrics(targets, probs)
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_loss = val_loss
                best_state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
                best_targets = targets
                best_probs = probs

        if best_state is None or best_targets is None or best_probs is None:
            raise RuntimeError(f"Fold {fold_idx} did not produce a valid checkpoint; verify the data balance and inputs.")

        torch.save(best_state, run_dir / f"fold{fold_idx}_best.pt")

        fpr, tpr, thresholds = roc_curve(best_targets, best_probs)
        save_fold_outputs(run_dir, fold_idx, best_targets, best_probs, fpr, tpr, thresholds)

        final_metrics = compute_metrics(best_targets, best_probs)
        fold_metrics.append(
            FoldMetrics(
                fold=fold_idx,
                accuracy=final_metrics["accuracy"],
                auc=final_metrics["auc"],
                f1=final_metrics["f1"],
                loss=best_loss,
            )
        )

    metrics_summary = summarize_folds(fold_metrics)
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as fp:
        json.dump({"folds": [m.to_dict() for m in fold_metrics], "summary": metrics_summary}, fp, indent=2)
    return metrics_summary


def holdout_experiment(config: ExperimentConfig) -> Dict[str, float]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volumes, labels = load_numpy_volumes(DATA_DIR)

    run_dir = DEFAULT_RUNS_ROOT / config.name
    write_metadata(run_dir, config)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=config.holdout_test_size, stratify=labels, random_state=config.seed
    )

    train_dataset = Knee3DDataset(volumes, labels, train_idx)
    test_dataset = Knee3DDataset(volumes, labels, test_idx)

    sampler = make_sampler(labels[train_idx])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(pretrained=config.pretrained, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_targets: np.ndarray | None = None
    best_probs: np.ndarray | None = None
    best_loss = float("inf")

    for _ in range(config.num_epochs):
        run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, targets, probs, _ = evaluate(model, test_loader, criterion, device)
        metrics = compute_metrics(targets, probs)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
            best_targets = targets
            best_probs = probs
            best_loss = val_loss

    if best_state is None or best_targets is None or best_probs is None:
        raise RuntimeError("Holdout experiment did not produce a valid checkpoint; check input data.")

    torch.save(best_state, run_dir / "holdout_best.pt")

    fpr, tpr, thresholds = roc_curve(best_targets, best_probs)
    save_fold_outputs(run_dir, 0, best_targets, best_probs, fpr, tpr, thresholds)

    holdout_metrics = {
        "accuracy": compute_metrics(best_targets, best_probs)["accuracy"],
        "auc": best_auc,
        "f1": compute_metrics(best_targets, best_probs)["f1"],
        "loss": best_loss,
    }
    with open(run_dir / "holdout_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(holdout_metrics, fp, indent=2)
    return holdout_metrics


