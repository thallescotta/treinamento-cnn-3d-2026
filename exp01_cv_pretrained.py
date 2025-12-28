"""Baseline 5-fold cross-validation with pretrained R3D-18 weights."""
from __future__ import annotations

from experiment_utils import ExperimentConfig, cross_validation_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        name="baseline_cv_pretrained",
        pretrained=True,
    )
    cross_validation_experiment(config)
