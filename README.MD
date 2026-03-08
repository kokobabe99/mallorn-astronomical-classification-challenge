# MALLORN Astronomical Classification Challenge

## Overview

This project tackles the [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge) Kaggle competition. The goal is to identify **Tidal Disruption Events (TDEs)** — stars torn apart by supermassive black holes — from simulated LSST multi-band light curves constructed from real ZTF observations.

**Competition Metric:** F1 Score (binary classification, target = 1 for TDE).

## Dataset

| Split | Objects | TDE (target=1) | Non-TDE (target=0) |
|-------|---------|-----------------|---------------------|
| Train | 3,043   | ~64             | ~2,979              |
| Test  | 7,135   | Unknown         | Unknown             |

The training set is **severely imbalanced** (~2% TDE). Non-TDE objects include AGN (Active Galactic Nuclei) and Type II Supernovae (SN II).

Each object has:
- **Metadata**: redshift (`Z`, `Z_err`), extinction (`EBV`), spectral type (`SpecType`)
- **Light curves**: multi-band (u, g, r, i, z, y) time-series photometry split across 20 data files, with columns `object_id`, `Time (MJD)`, `Filter`, `Flux`, `Flux_err`

## Data Parsing

The competition data is split into two parts: **metadata** and **light curves**.

### Metadata

Two CSV files at the top level:

- `train_log.csv` — 3,043 training objects with columns:
  - `object_id`: unique Tolkien-themed identifier (e.g. `Dornhoth_fervain_onodrim`)
  - `Z`, `Z_err`: redshift and its uncertainty (Z_err is missing for some objects)
  - `EBV`: dust extinction value
  - `SpecType`: spectral type — `AGN`, `SN II`, or `TDE`
  - `split`: which data split the light curve belongs to (`split_01` ... `split_20`)
  - `target`: binary label (1 = TDE, 0 = non-TDE)

- `test_log.csv` — 7,135 test objects with the same columns (except `target`)

### Light Curves

Light curve data is distributed across 20 subdirectories (`split_01/` to `split_20/`), each containing:
- `train_full_lightcurves.csv`
- `test_full_lightcurves.csv`

Each light curve file has columns:
- `object_id`: links back to the metadata
- `Time (MJD)`: observation timestamp in Modified Julian Date
- `Flux`: measured brightness
- `Flux_err`: flux measurement uncertainty
- `Filter`: LSST photometric band — one of `u`, `g`, `r`, `i`, `z`, `y`

Each split file contains ~26K (train) or ~59K (test) observation rows. To reconstruct a complete dataset, all 20 splits must be concatenated:

```python
train_log = pd.read_csv("data/train_log.csv")
test_log = pd.read_csv("data/test_log.csv")

train_lcs, test_lcs = [], []
for i in range(1, 21):
    d = f"data/split_{i:02d}"
    train_lcs.append(pd.read_csv(f"{d}/train_full_lightcurves.csv"))
    test_lcs.append(pd.read_csv(f"{d}/test_full_lightcurves.csv"))

train_lc = pd.concat(train_lcs, ignore_index=True)
test_lc = pd.concat(test_lcs, ignore_index=True)
```

Each `object_id` has multi-band time-series observations. To get a single object's light curve:

```python
obj_lc = train_lc[train_lc["object_id"] == "Dornhoth_fervain_onodrim"]
# Filter by band: obj_lc[obj_lc["Filter"] == "g"]
```

## Methodology

### 1. Feature Engineering

We extract **190+ features** from each object's light curve across several categories:

**Global Flux Statistics (16 features)**
- Mean, std, median, min, max, range, skewness, kurtosis
- Percentiles (P10, P90), IQR, MAD (median absolute deviation)
- Fraction of observations above mean, fraction with positive flux
- Inverse-variance weighted mean

**Error & Signal-to-Noise Features (6 features)**
- Flux error mean, std, median
- SNR mean, max, std

**Temporal Features (10 features)**
- Time span, cadence statistics (dt mean, std, min, max)
- Flux rate of change: mean, std, max, min, absolute max
- Fraction of positive flux rate (rising vs falling)

**Peak Analysis (7 features)**
- Peak flux, peak phase (relative timing), peak time from start
- Rise rate, rise duration, decline rate, decline duration
- Rise-to-decline duration ratio

**Variability Indices (4 features)**
- Stetson J index — detects correlated variability across consecutive observations
- Von Neumann ratio — measures time-series smoothness
- Excess variance — intrinsic variability beyond measurement noise
- Number of peaks (via `scipy.signal.find_peaks`)

**Per-Filter Features (17 features x 6 filters = 102 features)**
- For each of the 6 LSST bands (u, g, r, i, z, y): count, mean, std, max, min, range, skewness, kurtosis, SNR, weighted mean, MAD, P10, P90, positive fraction, max rate, mean rate, peak phase, Stetson J

**Color Features (18 features)**
- Mean flux differences between band pairs: (u,g), (g,r), (r,i), (i,z), (z,y), (g,i), (u,r), (r,z)
- Peak flux differences between the same band pairs
- Color variability: range differences in (g-r) and (r-i)

**Cross-Filter Timing (3 features)**
- Peak delay between bands: (g,r), (r,i), (i,z) — characteristic of physical processes with wavelength-dependent emission

**Metadata Interaction Features (8 features)**
- Flux mean/range/amplitude/peak divided by redshift
- Z × EBV, log(flux range), log(peak flux), has_z_err binary indicator

### 2. Model Ensemble

We use a **3-model ensemble** to maximize prediction diversity:

| Model | Type | Key Config |
|-------|------|------------|
| **LightGBM** | Gradient Boosting | lr=0.02, max_depth=7, 31 leaves, 3000 rounds |
| **XGBoost** | Gradient Boosting | lr=0.02, max_depth=7, 3000 rounds |
| **ExtraTrees** | Random Forest variant | 500 trees, max_depth=10, balanced class weights |

All models use `scale_pos_weight` (or `class_weight="balanced"`) to handle the severe class imbalance.

### 3. Cross-Validation

- **5-fold Stratified K-Fold** ensures each fold preserves the TDE class ratio
- Each model produces out-of-fold (OOF) predictions used for ensemble optimization
- Early stopping (patience=150 rounds) for LightGBM and XGBoost

### 4. Ensemble Weight & Threshold Optimization

After cross-validation, we perform a **grid search on OOF predictions** to find:
- **Optimal ensemble weights**: searched over `w_lgb`, `w_xgb`, `w_et` combinations (step=0.1)
- **Optimal classification threshold**: searched from 0.05 to 0.80 (step=0.005)

This jointly maximizes the F1 score on the full training set's OOF predictions.

## Experiments & Results

We explored multiple approaches, progressively improving the pipeline. All scores are F1 on Kaggle's private/public leaderboard.

### Approach 1: Baseline Ensemble (LightGBM + XGBoost + ExtraTrees)

- **Features**: ~150 hand-crafted features (global flux statistics, temporal features, variability indices, per-filter features, color features, cross-filter timing, metadata interactions)
- **Models**: LightGBM, XGBoost, ExtraTrees with 5-fold stratified CV
- **Ensemble**: Grid search over ensemble weights + threshold optimization on OOF predictions
- **No oversampling**

| Submission | Private Score | Public Score |
|---|---|---|
| Ensemble (weight-optimized) | 0.5268 | 0.5539 |

### Approach 2: Enhanced Features + Stacking + SMOTE

- **Features**: ~267 features — added Bazin function fits, structure function, autocorrelation, flux asymmetry, rolling statistics, Stetson K index
- **Models**: LightGBM, XGBoost, CatBoost, ExtraTrees, BalancedRandomForest (5-model)
- **SMOTE** oversampling (k=3) within each CV fold
- **Stacking**: Logistic Regression meta-learner on base model outputs
- **10-fold CV** with nested CV threshold selection

> This approach did not yield a clear improvement over Approach 1, likely due to feature noise from the larger feature set and potential overfitting with 5 models on a small TDE sample.

### Approach 3: Feature Selection + Optuna Tuning + Pseudo-Labeling

- **Features**: Top 60 features selected via multi-seed LightGBM importance (seeds: 42, 123, 456) — reduced noise from the 267-feature set
- **Models**: LightGBM, XGBoost, CatBoost (focused 3-model ensemble, dropped weaker models)
- **Optuna** Bayesian hyperparameter tuning: 50 trials per model
- **SMOTE** (k=3) within 10-fold CV
- **Pseudo-Labeling**: 2 rounds of self-training on high-confidence test predictions
  - Round 1: pos ≥ 0.90, neg ≤ 0.05
  - Round 2: pos ≥ 0.85, neg ≤ 0.08
- **Nested CV** threshold selection (median from 5-fold inner CV)

| Submission | Private Score | Public Score |
|---|---|---|
| **3-Model Ensemble** | **0.5481** | **0.5643** |
| **Pseudo-Label Round 2** | **0.5481** | **0.5643** |
| Pseudo-Label Round 1 | 0.5443 | 0.5505 |
| XGBoost (single model) | 0.5358 | 0.5438 |
| LightGBM (single model) | 0.5230 | 0.5340 |
| Base Ensemble (nested CV threshold) | 0.5212 | 0.5340 |
| CatBoost (single model) | 0.4664 | 0.5073 |

### Approach 4: MLP + Multi-Seed Blending

- **Features**: Same top 60 features as Approach 3, with StandardScaler for MLP
- **Models**: LightGBM, XGBoost, CatBoost, **MLP** (scikit-learn MLPClassifier, 2-3 hidden layers)
- **Multi-seed blending**: Each model trained with 3 seeds (42, 123, 456) × 10-fold CV, predictions averaged across seeds for stability
- **Optuna** tuning for all 4 models (40-50 trials each)
- **Pseudo-Labeling**: 2 rounds (same thresholds as Approach 3)

| Submission | Private Score | Public Score |
|---|---|---|
| MLP (single model) | 0.4573 | 0.4545 |

> MLP underperformed tree-based models significantly. Multi-seed blending added training cost without improving over Approach 3's focused ensemble.

### Summary

| Approach | Best Private Score | Key Technique |
|---|---|---|
| Baseline Ensemble | 0.5268 | Weight-optimized 3-model ensemble |
| Enhanced Features + Stacking | ~0.5268 | SMOTE + 5-model stacking |
| **Feature Selection + Optuna + PL** | **0.5481** | Feature selection + HPO + pseudo-labeling |
| MLP + Multi-Seed | 0.4573 | Neural network + multi-seed blending |

**Best result: F1 = 0.5481** (Private) achieved by the Approach 3 ensemble and its pseudo-labeled variant.

### Key Takeaways

- **Feature selection > feature quantity**: Reducing from 267 to 60 features improved generalization
- **Hyperparameter tuning matters**: Optuna-tuned models significantly outperformed manually-tuned ones
- **Pseudo-labeling helped slightly**: Round 2 matched the base ensemble; Round 1 was close
- **Tree-based models dominate**: XGBoost was the strongest single model; MLP was the weakest
- **CatBoost underperformed** relative to LightGBM and XGBoost on this task
- **Threshold optimization is critical** for imbalanced classification — optimal threshold was far below 0.5

## Project Structure

```
mallorn-astronomical-classification-challenge/
├── data/
│   ├── train_log.csv                    # Train metadata (3,043 objects)
│   ├── test_log.csv                     # Test metadata (7,135 objects)
│   ├── sample_submission.csv            # Expected submission format
│   └── split_01/ ... split_20/          # Light curve data
│       ├── train_full_lightcurves.csv
│       └── test_full_lightcurves.csv
├── mallorn_classification.ipynb         # Approach 1: Baseline ensemble
├── mallorn_classification_v2.ipynb      # Approach 2: Enhanced features + stacking
├── mallorn_classification_v3.ipynb      # Approach 3: Feature selection + Optuna + PL
├── mallorn_classification_v4.ipynb      # Approach 4: MLP + multi-seed blending
├── requirements.txt                     # Python dependencies
└── README.MD
```

## How to Run

Upload and run any notebook on Kaggle with the competition dataset attached. Each notebook includes EDA, full training pipeline, and submission generation.

## Dependencies

- Python 3.12
- pandas, numpy, scipy
- scikit-learn
- lightgbm
- xgboost
- matplotlib
- tqdm

## Analysis

- **Class imbalance is the central challenge**: only ~2% of training objects are TDEs. Standard accuracy metrics would be misleading; F1 score focuses on the minority class. Threshold optimization on OOF predictions is essential.
- **TDEs have distinctive light curve morphology**: rapid flux rise followed by a power-law decline, which differs from the stochastic variability of AGN and the plateau behavior of SN II. Features capturing rise/decline asymmetry and variability indices are among the most important.
- **Multi-band color evolution matters**: TDEs show characteristic blue color that evolves over time, captured by cross-filter peak delays and color features.
- **Redshift normalization is physically motivated**: dividing flux-based features by redshift accounts for distance effects, making features more comparable across objects at different distances.
- **Ensemble diversity improves robustness**: LightGBM and XGBoost learn similar gradient-boosted patterns but differ in regularization and splitting strategies, while ExtraTrees provides a complementary randomized approach that reduces overfitting on the small TDE sample.
