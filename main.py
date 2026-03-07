"""
MALLORN Astronomical Classification Challenge
Goal: Identify Tidal Disruption Events (TDEs) from simulated LSST light curves.
Metric: F1 Score (binary classification, target=1 for TDE)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FILTERS = ["u", "g", "r", "i", "z", "y"]
N_SPLITS = 20


# =============================================================================
# 1. Load Data
# =============================================================================
def load_metadata():
    """Load train and test metadata logs."""
    train_log = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))
    test_log = pd.read_csv(os.path.join(DATA_DIR, "test_log.csv"))
    print(f"Train: {len(train_log)} objects ({train_log['target'].sum()} TDEs)")
    print(f"Test:  {len(test_log)} objects")
    return train_log, test_log


def load_all_lightcurves():
    """Load and concatenate light curves from all 20 splits."""
    train_lcs = []
    test_lcs = []
    for i in tqdm(range(1, N_SPLITS + 1), desc="Loading light curves"):
        split_dir = os.path.join(DATA_DIR, f"split_{i:02d}")
        train_lcs.append(pd.read_csv(os.path.join(split_dir, "train_full_lightcurves.csv")))
        test_lcs.append(pd.read_csv(os.path.join(split_dir, "test_full_lightcurves.csv")))
    train_lc = pd.concat(train_lcs, ignore_index=True)
    test_lc = pd.concat(test_lcs, ignore_index=True)
    print(f"Train LC rows: {len(train_lc)}, Test LC rows: {len(test_lc)}")
    return train_lc, test_lc


# =============================================================================
# 2. Feature Engineering
# =============================================================================
def extract_features_for_object(group):
    """Extract statistical and temporal features from one object's light curve."""
    feats = {}

    flux = group["Flux"].values
    flux_err = group["Flux_err"].values
    time = group["Time (MJD)"].values

    # --- Global features ---
    feats["n_obs"] = len(flux)
    feats["flux_mean"] = np.mean(flux)
    feats["flux_std"] = np.std(flux)
    feats["flux_median"] = np.median(flux)
    feats["flux_min"] = np.min(flux)
    feats["flux_max"] = np.max(flux)
    feats["flux_range"] = feats["flux_max"] - feats["flux_min"]
    feats["flux_skew"] = stats.skew(flux) if len(flux) > 2 else 0
    feats["flux_kurtosis"] = stats.kurtosis(flux) if len(flux) > 3 else 0
    feats["flux_iqr"] = np.percentile(flux, 75) - np.percentile(flux, 25)
    feats["flux_above_mean_frac"] = np.mean(flux > feats["flux_mean"])

    # Flux error stats
    feats["flux_err_mean"] = np.mean(flux_err)
    feats["flux_err_std"] = np.std(flux_err)
    feats["snr_mean"] = np.mean(np.abs(flux) / (flux_err + 1e-10))

    # Time span
    feats["time_span"] = np.ptp(time) if len(time) > 1 else 0

    # Sorted by time for temporal features
    sort_idx = np.argsort(time)
    flux_sorted = flux[sort_idx]
    time_sorted = time[sort_idx]

    if len(flux_sorted) > 1:
        dt = np.diff(time_sorted)
        dflux = np.diff(flux_sorted)
        rates = dflux / (dt + 1e-10)
        feats["flux_rate_mean"] = np.mean(rates)
        feats["flux_rate_std"] = np.std(rates)
        feats["flux_rate_max"] = np.max(np.abs(rates))
        feats["dt_mean"] = np.mean(dt)
        feats["dt_std"] = np.std(dt)
    else:
        feats["flux_rate_mean"] = 0
        feats["flux_rate_std"] = 0
        feats["flux_rate_max"] = 0
        feats["dt_mean"] = 0
        feats["dt_std"] = 0

    # Peak features: position of max flux relative to time span
    if len(flux_sorted) > 1:
        peak_idx = np.argmax(flux_sorted)
        feats["peak_phase"] = (time_sorted[peak_idx] - time_sorted[0]) / (feats["time_span"] + 1e-10)
    else:
        feats["peak_phase"] = 0.5

    # Amplitude / noise ratio
    feats["amplitude_snr"] = feats["flux_range"] / (feats["flux_err_mean"] + 1e-10)

    # --- Per-filter features ---
    for filt in FILTERS:
        mask = group["Filter"].values == filt
        f_flux = flux[mask]
        f_err = flux_err[mask]

        prefix = f"f_{filt}_"
        feats[prefix + "n"] = len(f_flux)
        if len(f_flux) > 0:
            feats[prefix + "mean"] = np.mean(f_flux)
            feats[prefix + "std"] = np.std(f_flux)
            feats[prefix + "max"] = np.max(f_flux)
            feats[prefix + "min"] = np.min(f_flux)
            feats[prefix + "range"] = feats[prefix + "max"] - feats[prefix + "min"]
            feats[prefix + "skew"] = stats.skew(f_flux) if len(f_flux) > 2 else 0
            feats[prefix + "snr"] = np.mean(np.abs(f_flux) / (f_err + 1e-10))
        else:
            feats[prefix + "mean"] = 0
            feats[prefix + "std"] = 0
            feats[prefix + "max"] = 0
            feats[prefix + "min"] = 0
            feats[prefix + "range"] = 0
            feats[prefix + "skew"] = 0
            feats[prefix + "snr"] = 0

    # --- Color features (flux ratios between bands) ---
    band_means = {}
    for filt in FILTERS:
        mask = group["Filter"].values == filt
        f_flux = flux[mask]
        band_means[filt] = np.mean(f_flux) if len(f_flux) > 0 else 0

    color_pairs = [("g", "r"), ("r", "i"), ("i", "z"), ("g", "i"), ("u", "g"), ("z", "y")]
    for b1, b2 in color_pairs:
        feats[f"color_{b1}_{b2}"] = band_means[b1] - band_means[b2]

    # --- Variability features per filter ---
    for filt in FILTERS:
        mask = group["Filter"].values == filt
        f_flux = flux[mask]
        f_time = time[mask]
        prefix = f"f_{filt}_"
        if len(f_flux) > 2:
            sort_i = np.argsort(f_time)
            f_sorted = f_flux[sort_i]
            t_sorted = f_time[sort_i]
            dt_f = np.diff(t_sorted)
            df_f = np.diff(f_sorted)
            rates_f = df_f / (dt_f + 1e-10)
            feats[prefix + "rate_max"] = np.max(np.abs(rates_f))
            feats[prefix + "rate_mean"] = np.mean(rates_f)
        else:
            feats[prefix + "rate_max"] = 0
            feats[prefix + "rate_mean"] = 0

    return feats


def build_features(lc_df, meta_df, desc="Building features"):
    """Build feature matrix from light curves + metadata."""
    # Group light curves by object
    grouped = lc_df.groupby("object_id")

    records = []
    for obj_id, group in tqdm(grouped, desc=desc):
        feats = extract_features_for_object(group)
        feats["object_id"] = obj_id
        records.append(feats)

    feat_df = pd.DataFrame(records)

    # Merge with metadata
    merged = meta_df.merge(feat_df, on="object_id", how="left")

    # Use Z (redshift) and EBV as features
    merged["Z"] = pd.to_numeric(merged["Z"], errors="coerce")
    merged["Z_err"] = pd.to_numeric(merged["Z_err"], errors="coerce")
    merged["EBV"] = pd.to_numeric(merged["EBV"], errors="coerce")

    # Interaction features
    merged["flux_mean_over_z"] = merged["flux_mean"] / (merged["Z"] + 1e-10)
    merged["flux_range_over_z"] = merged["flux_range"] / (merged["Z"] + 1e-10)

    return merged


# =============================================================================
# 3. Model Training & Prediction
# =============================================================================
def get_feature_columns(df):
    """Get numeric feature columns (exclude identifiers and target)."""
    exclude = {"object_id", "SpecType", "English Translation", "split", "target"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


def train_and_predict(train_df, test_df):
    """Train LightGBM + XGBoost ensemble with cross-validation."""
    feature_cols = get_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features")

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["target"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale ratio for class imbalance
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos = neg_count / max(pos_count, 1)
    print(f"Class ratio: {neg_count}:{pos_count}, scale_pos_weight={scale_pos:.1f}")

    # --- LightGBM ---
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 5,
        "scale_pos_weight": scale_pos,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    # --- XGBoost ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "scale_pos_weight": scale_pos,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
        "nthread": -1,
        "random_state": 42,
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        # XGBoost
        xgb_train = xgb.DMatrix(X_tr, label=y_tr)
        xgb_val = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            xgb_params,
            xgb_train,
            num_boost_round=2000,
            evals=[(xgb_val, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        # Predict validation
        val_pred_lgb = lgb_model.predict(X_val)
        val_pred_xgb = xgb_model.predict(xgb.DMatrix(X_val))
        val_pred_ensemble = 0.5 * val_pred_lgb + 0.5 * val_pred_xgb
        oof_preds[val_idx] = val_pred_ensemble

        # Find best threshold for F1
        best_f1, best_thr = 0, 0.5
        for thr in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y_val, (val_pred_ensemble >= thr).astype(int))
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        f1_scores.append(best_f1)
        print(f"  Fold {fold+1}: F1={best_f1:.4f} (threshold={best_thr:.2f})")

        # Predict test
        test_preds_lgb += lgb_model.predict(X_test) / skf.n_splits
        test_preds_xgb += xgb_model.predict(xgb.DMatrix(X_test)) / skf.n_splits

    # Ensemble
    test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb

    print(f"\nCV Mean F1: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

    # Optimize global threshold on OOF
    best_f1_global, best_thr_global = 0, 0.5
    for thr in np.arange(0.05, 0.9, 0.005):
        f1 = f1_score(y, (oof_preds >= thr).astype(int))
        if f1 > best_f1_global:
            best_f1_global, best_thr_global = f1, thr

    print(f"Best OOF F1: {best_f1_global:.4f} at threshold={best_thr_global:.3f}")
    print(classification_report(y, (oof_preds >= best_thr_global).astype(int), target_names=["Non-TDE", "TDE"]))

    # Feature importance
    lgb_final = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": 1000})
    lgb_final.fit(X, y)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgb_final.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))

    return test_preds, best_thr_global


# =============================================================================
# 4. Main Pipeline
# =============================================================================
def main():
    print("=" * 60)
    print("MALLORN Astronomical Classification Challenge")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading metadata...")
    train_log, test_log = load_metadata()

    print("\n[2/4] Loading light curves...")
    train_lc, test_lc = load_all_lightcurves()

    # Feature engineering
    print("\n[3/4] Feature engineering...")
    train_feat = build_features(train_lc, train_log, desc="Train features")
    test_feat = build_features(test_lc, test_log, desc="Test features")

    # Train & predict
    print("\n[4/4] Training models...")
    test_probs, threshold = train_and_predict(train_feat, test_feat)

    # Generate submission
    submission = pd.DataFrame({
        "object_id": test_log["object_id"],
        "prediction": (test_probs >= threshold).astype(int),
    })

    # Sanity check
    print(f"\nSubmission: {len(submission)} rows")
    print(f"Predicted TDEs: {submission['prediction'].sum()}")

    submission.to_csv("submission.csv", index=False)
    print("Saved to submission.csv")

    # Verify format matches sample
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    assert list(submission.columns) == list(sample.columns), "Column mismatch!"
    assert len(submission) == len(sample), "Row count mismatch!"
    print("Submission format verified OK!")


if __name__ == "__main__":
    main()
