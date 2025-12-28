"""Holdout evaluation and ROC/AUC consistency check."""
from __future__ import annotations

from experiment_utils import ExperimentConfig, holdout_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        name="holdout_final_roc",
        pretrained=True,
    )
    holdout_experiment(config)
