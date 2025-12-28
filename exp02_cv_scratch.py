"""Ablation study: 5-fold cross-validation training from scratch."""
from __future__ import annotations

from experiment_utils import ExperimentConfig, cross_validation_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        name="ablation_cv_scratch",
        pretrained=False,
    )
    cross_validation_experiment(config)
