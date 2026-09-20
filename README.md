# treinamento-cnn-3d-2026

3D CNN (R3D-18) pipeline with stratified 5-fold cross-validation for binary classification (normal × abnormal) of knee 3D DESS MRI volumes from the Osteoarthritis Initiative (OAI).

> **Status:** work in progress. This repository currently contains only Experiment 01 (`new_ex1.py`). See [Known limitations](#known-limitations-of-this-version).

---

## Requirements

- **Python:** 3.11 (used in the 2026 experiments).
- **Packages:** `torch`, `torchvision`, `numpy`, `scikit-learn`, `imbalanced-learn`, `tqdm`.
- **GPU:** an NVIDIA GPU with CUDA is recommended. Without a GPU the script falls back to CPU with batch size 2 and AMP disabled, which is very slow.

```bash
# create and activate a virtual environment, then:
pip install torch torchvision numpy scikit-learn imbalanced-learn tqdm
```

Install the PyTorch build that matches your CUDA version, following <https://pytorch.org/get-started/locally/>. Exact package versions are not pinned yet; to record your own environment, run `pip freeze`.

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Overview

The pipeline trains and evaluates **R3D-18**, pretrained on Kinetics-400, on knee MRI volumes (3D DESS sequence, OAI) using **stratified 5-fold cross-validation**.

The code was adapted to run on **Windows 11 with a single GPU**, because the Linux environment with two GPUs (LASAI, CEFET/RJ) used in earlier experiments was reformatted and access for reproduction there was not restored.

---

## Repository contents

| File | Description | Status |
|---|---|---|
| `new_ex1.py` | Experiment 01: stratified 5-fold CV with pretrained R3D-18 | available |
| `exp02_cv_scratch.py` | Experiment 02: R3D-18 trained from scratch (comparison with pretraining) | **not yet included** |
| `exp03_holdout_final_roc.py` | Experiment 03: holdout evaluation and ROC curve | **not yet included** |

Results (models, metrics, and logs) are **not** versioned here. They are written locally to `C:\dataset\runs` (see [Outputs](#outputs)).

---

## Relation to publications

| Publication | DOI | Code in this repository? |
|---|---|---|
| Fontainha et al., *Classificação Binária de Imagens de Ressonância Magnética de Osteoartrite com o Modelo R3D_18 Modificado* (English title: *Binary Classification of MRI Images of Osteoarthritis with a Modified R3D_18 Model*), SBCAS 2025, pp. 1017–1022. Short paper; Best Paper Award in the Short Papers category of the Main Track. | [10.5753/sbcas.2025.7629](https://doi.org/10.5753/sbcas.2025.7629) | **No.** Code for the 2025 experiments: [TO FILL IN: location, or "not available"] |
| Fontainha et al., *Avaliação de Técnicas de Aumento de Dados para Classificação Binária de Imagens Volumétricas de Ressonância Magnética do Joelho* (free translation: *Evaluation of Data Augmentation Techniques for Binary Classification of Volumetric Knee MRI*), SBCAS 2026, pp. 217–228. | [10.5753/sbcas.2026.20619](https://doi.org/10.5753/sbcas.2026.20619) | **No.** Code for experiments EX1–EX10: [TO FILL IN: location, or "not available"] |

`new_ex1.py` is the 2026 single-GPU pipeline. **Its configuration does not match the one described in the SBCAS 2026 paper.** Here oversampling is enabled, the batch size is 16, training runs for at most 50 epochs, and there is no external test set. The paper describes batch size 32, up to 60 epochs, no oversampling, and an isolated external test set. Do not use this script as a reproduction of the numbers in that paper.

---

## Dataset

Knee MRI volumes (3D DESS) from the **Osteoarthritis Initiative (OAI)**, in the NumPy-preprocessed version available on Kaggle:

- Kaggle: <https://www.kaggle.com/datasets/mohamedberrimi/oaimri3ddess/data>
- OAI (NIMH Data Archive): <https://nda.nih.gov/oai>

Files expected in the data directory (default `C:\dataset\data`):

- `normal-3DESS-128-64.npy`: class 0 (normal).
- `abnormal-3DESS-128-64.npy`: class 1 (abnormal).

Each sample is a 128 × 128 × 64 volume. The label is defined by the source file. The arrays **do not contain patient identifiers**.

The data are not redistributed in this repository (size and licensing). Use of the data is subject to the terms of OAI/NDA and of the Kaggle dataset page. Class counts are printed at run time (`Dataset carregado: N normais, M anormais`) and saved to `experiment_summary.json` (field `dataset_info`).

---

## Protocol implemented in `new_ex1.py`

- **Partitioning:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over all volumes (volume-level split).
- **Oversampling:** `RandomOverSampler(random_state=42 + fold)` applied only to the training indices of each fold.
- **Model:** `torchvision.models.video.r3d_18` with Kinetics-400 weights (`R3D_18_Weights.DEFAULT`). The final layer is replaced by `Linear(in_features, 1)` (Xavier initialization). The single input channel is repeated into 3 channels.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight = n_normal / n_abnormal`, computed over the whole dataset.
- **Optimization:** Adam, learning rate 1e-4, weight decay 1e-5; batch size 16; automatic mixed precision (AMP); up to 50 epochs.
- **Early stopping and checkpointing:** patience 15, monitoring the **validation-fold loss**. The model from the epoch with the lowest validation loss is saved, and the fold metrics are computed from it.
- **Preprocessing:** per-volume min-max normalization; random flips (p = 0.5 each) along the last two axes of the volume tensor.
- **Reproducibility:** fixed seeds (Python, NumPy, PyTorch), `cudnn.deterministic=True`, `cudnn.benchmark=False`. GPU execution may still vary across machines.
- **Per-fold metrics:** accuracy, AUC-ROC, F1, AUC-PR, sensitivity, specificity, precision (PPV), NPV, and confusion matrix (threshold 0.5). Aggregation: mean and standard deviation across folds (`np.std`, ddof = 0) and global metrics over the concatenated predictions.

---

## Running the experiment

1. Place the two `.npy` files in the data directory.
2. Edit the `Config` class at the top of `new_ex1.py` (`DATA_DIR`, `OUTPUT_DIR`, `EXP_NAME`, hyperparameters). **The script does not accept command-line arguments.**
3. Run:

```bash
python new_ex1.py
```

### Environment (2026 experiments)

- Windows 11; NVIDIA RTX 5050 GPU (8 GB VRAM); 16 GB RAM.
- Python 3.11; PyTorch, Torchvision, NumPy, scikit-learn, imbalanced-learn, tqdm.

Earlier work used a Linux environment with two NVIDIA RTX 2080 Ti GPUs (12 GB each), which is no longer available for reproduction.

---

## Outputs

Written to `OUTPUT_DIR/EXP_NAME/` (default `C:\dataset\runs\exp01_cv_pretrained\`):

- `config.json`, `log_YYYYMMDD_HHMMSS.txt`, `experiment_summary.json`.
- `fold_1/` … `fold_5/`: `best_model.pth`, `y_true.npy`, `y_prob.npy`, `roc.json`, `training_history.json`, `metrics.json`, `fold_result.json`.
- `all_labels.npy`, `all_probs.npy`, `global_roc.json`: concatenated predictions from the 5 folds.

The script also prints a LaTeX table with the per-fold results.

---

## Known limitations of this version

1. **Volume-level split.** Without patient identifiers in the arrays, it cannot be guaranteed that volumes from the same patient stay in the same partition; performance estimates may be inflated.
2. **No held-out test set.** Per-fold metrics come from the validation fold, which is also used to choose the epoch (early stopping). They are slightly optimistic estimates, not performance on unseen data.
3. **Data augmentation also applied to validation.** The same `Dataset` object (with random flips) feeds both the training and validation loaders.
4. **`pos_weight` combined with oversampling.** The weight is computed over the whole dataset and added on top of oversampling, which gives extra weight to the positive class.
5. **Dead code:** `RandomRotation3D` does not rotate and is not used; `LOSS_TYPE` is never read.
6. **Hard-coded paths** (`C:\dataset\...`) defined in `Config`.

---

## Next steps

- Add the scripts for Experiments 02 and 03 and the results (`experiment_summary.json`, per-fold predictions).
- In Experiment 03, separate the test set **before** cross-validation and use CV only on the development set; otherwise the test set is contaminated.
- Add a `LICENSE` file and publish a release with a DOI (for example, via Zenodo).

---

## Author

Thalles Cotta Fontainha — Postgraduate Program in Instrumentation and Applied Optics, CEFET/RJ — thalles.fontainha@aluno.cefet-rj.br

Co-authors of the publications: F. da R. Henriques, A. A. Lima, G. M. Araujo, R. de S. Tesch.

## License

[TO FILL IN: code license (for example, MIT or Apache-2.0), and create the `LICENSE` file.] OAI/Kaggle data follow their own terms of use.
