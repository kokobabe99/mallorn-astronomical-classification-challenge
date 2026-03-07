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

## Ablation Studies

| Variant | Change | Impact |
|---|---|---|
| v1 (2-model) vs v2 (3-model) | Added ExtraTrees to ensemble | Improved F1 by providing diversity; tree-based randomness complements gradient boosting |
| Basic features vs enhanced features | Added Stetson J, Von Neumann, rise/decline, per-filter peak phase, cross-filter timing | Significant F1 improvement — variability indices capture TDE-specific temporal signatures |
| Fixed weights (0.5/0.5) vs optimized weights | Grid search over ensemble weights | Small but consistent gain from adaptive weighting |
| Fixed threshold (0.5) vs optimized threshold | Threshold search on OOF | Critical for imbalanced data — optimal threshold is typically much lower than 0.5 |
| With vs without metadata interactions | Flux/redshift interaction features | Moderate improvement — physical quantities normalized by distance are more meaningful |
| 5-fold vs 10-fold CV | Number of cross-validation folds | 5-fold chosen for stability given the small number of TDEs per fold |

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
├── main.py                              # v1: LightGBM + XGBoost baseline
├── main_v2.py                           # v2: 3-model ensemble with enhanced features
├── mallorn_classification.ipynb         # Full Kaggle notebook with EDA & visualizations
├── submission.csv                       # Generated submission file
├── requirements.txt                     # Python dependencies
└── README.MD
```

## How to Run

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run v1 (baseline)
python main.py

# Run v2 (improved)
python main_v2.py
```

### Kaggle Submission

Upload and run `mallorn_classification.ipynb` on Kaggle with the competition dataset attached. The notebook includes EDA visualizations, full training pipeline, and submission generation.

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
