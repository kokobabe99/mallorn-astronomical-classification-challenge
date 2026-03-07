"""
MALLORN Astronomical Classification Challenge - v2 (Improved)
Key improvements:
  - Advanced light curve features (rise/decline rates, periodicity, percentiles)
  - Per-filter temporal features (rise time, decay time, peak flux)
  - Better feature interactions
  - Optimized hyperparameters
  - 3-model ensemble (LightGBM + XGBoost + Extra Trees)
  - Threshold search on OOF predictions
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier
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
    train_log = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))
    test_log = pd.read_csv(os.path.join(DATA_DIR, "test_log.csv"))
    print(f"Train: {len(train_log)} objects ({train_log['target'].sum()} TDEs)")
    print(f"Test:  {len(test_log)} objects")
    return train_log, test_log


def load_all_lightcurves():
    train_lcs, test_lcs = [], []
    for i in tqdm(range(1, N_SPLITS + 1), desc="Loading splits"):
        d = os.path.join(DATA_DIR, f"split_{i:02d}")
        train_lcs.append(pd.read_csv(os.path.join(d, "train_full_lightcurves.csv")))
        test_lcs.append(pd.read_csv(os.path.join(d, "test_full_lightcurves.csv")))
    return pd.concat(train_lcs, ignore_index=True), pd.concat(test_lcs, ignore_index=True)


# =============================================================================
# 2. Feature Engineering (Enhanced)
# =============================================================================
def safe_stat(arr, func, default=0):
    try:
        if len(arr) < 2:
            return default
        val = func(arr)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def compute_weighted_mean(flux, flux_err):
    """Inverse-variance weighted mean."""
    w = 1.0 / (flux_err**2 + 1e-10)
    return np.sum(flux * w) / np.sum(w)


def compute_stetson_j(flux, flux_err):
    """Stetson J variability index (simplified for single band)."""
    if len(flux) < 3:
        return 0
    mean = compute_weighted_mean(flux, flux_err)
    residuals = (flux - mean) / (flux_err + 1e-10)
    n = len(flux)
    pairs = residuals[:-1] * residuals[1:]
    sign_pairs = np.sign(pairs)
    return np.sum(sign_pairs * np.sqrt(np.abs(pairs))) / n


def compute_von_neumann_ratio(flux):
    """Von Neumann ratio - measures smoothness of time series."""
    if len(flux) < 3:
        return 0
    var = np.var(flux)
    if var < 1e-10:
        return 0
    delta_sq = np.mean(np.diff(flux)**2)
    return delta_sq / var


def extract_features_for_object(group):
    feats = {}
    flux = group["Flux"].values
    flux_err = group["Flux_err"].values
    time = group["Time (MJD)"].values
    filters = group["Filter"].values

    sort_idx = np.argsort(time)
    flux_s = flux[sort_idx]
    time_s = time[sort_idx]
    flux_err_s = flux_err[sort_idx]
    filters_s = filters[sort_idx]

    # --- Global basic stats ---
    n = len(flux)
    feats["n_obs"] = n
    feats["flux_mean"] = np.mean(flux)
    feats["flux_std"] = np.std(flux) if n > 1 else 0
    feats["flux_median"] = np.median(flux)
    feats["flux_min"] = np.min(flux)
    feats["flux_max"] = np.max(flux)
    feats["flux_range"] = feats["flux_max"] - feats["flux_min"]
    feats["flux_skew"] = safe_stat(flux, stats.skew)
    feats["flux_kurtosis"] = safe_stat(flux, stats.kurtosis)
    feats["flux_iqr"] = np.percentile(flux, 75) - np.percentile(flux, 25)
    feats["flux_p10"] = np.percentile(flux, 10)
    feats["flux_p90"] = np.percentile(flux, 90)
    feats["flux_above_mean_frac"] = np.mean(flux > feats["flux_mean"])
    feats["flux_positive_frac"] = np.mean(flux > 0)

    # Robust scatter
    feats["flux_mad"] = np.median(np.abs(flux - feats["flux_median"]))

    # Weighted mean
    feats["flux_wmean"] = compute_weighted_mean(flux, flux_err)

    # --- Error features ---
    feats["flux_err_mean"] = np.mean(flux_err)
    feats["flux_err_std"] = np.std(flux_err) if n > 1 else 0
    feats["flux_err_median"] = np.median(flux_err)
    feats["snr_mean"] = np.mean(np.abs(flux) / (flux_err + 1e-10))
    feats["snr_max"] = np.max(np.abs(flux) / (flux_err + 1e-10))
    feats["snr_std"] = np.std(np.abs(flux) / (flux_err + 1e-10)) if n > 1 else 0

    # --- Time features ---
    feats["time_span"] = np.ptp(time_s) if n > 1 else 0

    if n > 1:
        dt = np.diff(time_s)
        feats["dt_mean"] = np.mean(dt)
        feats["dt_std"] = np.std(dt)
        feats["dt_min"] = np.min(dt)
        feats["dt_max"] = np.max(dt)

        dflux = np.diff(flux_s)
        rates = dflux / (dt + 1e-10)
        feats["flux_rate_mean"] = np.mean(rates)
        feats["flux_rate_std"] = np.std(rates)
        feats["flux_rate_max"] = np.max(rates)
        feats["flux_rate_min"] = np.min(rates)
        feats["flux_rate_abs_max"] = np.max(np.abs(rates))
        feats["flux_rate_pos_frac"] = np.mean(rates > 0)
    else:
        for k in ["dt_mean", "dt_std", "dt_min", "dt_max",
                   "flux_rate_mean", "flux_rate_std", "flux_rate_max",
                   "flux_rate_min", "flux_rate_abs_max", "flux_rate_pos_frac"]:
            feats[k] = 0

    # --- Peak analysis ---
    peak_idx = np.argmax(flux_s)
    feats["peak_flux"] = flux_s[peak_idx]
    feats["peak_phase"] = (time_s[peak_idx] - time_s[0]) / (feats["time_span"] + 1e-10) if n > 1 else 0.5
    feats["peak_time_from_start"] = time_s[peak_idx] - time_s[0]

    # Rise and decline behavior
    if peak_idx > 0:
        rise_flux = flux_s[:peak_idx + 1]
        rise_time = time_s[:peak_idx + 1]
        feats["rise_rate"] = (rise_flux[-1] - rise_flux[0]) / (rise_time[-1] - rise_time[0] + 1e-10)
        feats["rise_duration"] = rise_time[-1] - rise_time[0]
    else:
        feats["rise_rate"] = 0
        feats["rise_duration"] = 0

    if peak_idx < n - 1:
        dec_flux = flux_s[peak_idx:]
        dec_time = time_s[peak_idx:]
        feats["decline_rate"] = (dec_flux[-1] - dec_flux[0]) / (dec_time[-1] - dec_time[0] + 1e-10)
        feats["decline_duration"] = dec_time[-1] - dec_time[0]
    else:
        feats["decline_rate"] = 0
        feats["decline_duration"] = 0

    feats["rise_decline_ratio"] = feats["rise_duration"] / (feats["decline_duration"] + 1e-10)
    feats["amplitude_snr"] = feats["flux_range"] / (feats["flux_err_mean"] + 1e-10)

    # --- Variability indices ---
    feats["stetson_j"] = compute_stetson_j(flux_s, flux_err_s)
    feats["von_neumann"] = compute_von_neumann_ratio(flux_s)

    # Excess variance
    mean_err_sq = np.mean(flux_err**2)
    feats["excess_variance"] = (np.var(flux) - mean_err_sq) / (feats["flux_mean"]**2 + 1e-10)

    # Number of peaks in the light curve
    if n > 5:
        try:
            peaks, _ = find_peaks(flux_s, height=feats["flux_mean"])
            feats["n_peaks"] = len(peaks)
        except Exception:
            feats["n_peaks"] = 0
    else:
        feats["n_peaks"] = 0

    # --- Per-filter features ---
    band_means = {}
    band_peak_flux = {}
    for filt in FILTERS:
        mask = filters == filt
        f_flux = flux[mask]
        f_err = flux_err[mask]
        f_time = time[mask]

        prefix = f"f_{filt}_"
        feats[prefix + "n"] = len(f_flux)

        if len(f_flux) > 0:
            feats[prefix + "mean"] = np.mean(f_flux)
            feats[prefix + "std"] = np.std(f_flux) if len(f_flux) > 1 else 0
            feats[prefix + "max"] = np.max(f_flux)
            feats[prefix + "min"] = np.min(f_flux)
            feats[prefix + "range"] = feats[prefix + "max"] - feats[prefix + "min"]
            feats[prefix + "skew"] = safe_stat(f_flux, stats.skew)
            feats[prefix + "kurtosis"] = safe_stat(f_flux, stats.kurtosis)
            feats[prefix + "snr"] = np.mean(np.abs(f_flux) / (f_err + 1e-10))
            feats[prefix + "wmean"] = compute_weighted_mean(f_flux, f_err)
            feats[prefix + "mad"] = np.median(np.abs(f_flux - np.median(f_flux)))
            feats[prefix + "p10"] = np.percentile(f_flux, 10)
            feats[prefix + "p90"] = np.percentile(f_flux, 90)
            feats[prefix + "positive_frac"] = np.mean(f_flux > 0)

            band_means[filt] = feats[prefix + "mean"]
            band_peak_flux[filt] = feats[prefix + "max"]

            # Per-filter temporal features
            if len(f_flux) > 2:
                si = np.argsort(f_time)
                fs = f_flux[si]
                ts = f_time[si]
                dt_f = np.diff(ts)
                df_f = np.diff(fs)
                rates_f = df_f / (dt_f + 1e-10)
                feats[prefix + "rate_max"] = np.max(np.abs(rates_f))
                feats[prefix + "rate_mean"] = np.mean(rates_f)

                # Peak timing in this band
                pk = np.argmax(fs)
                feats[prefix + "peak_phase"] = (ts[pk] - ts[0]) / (ts[-1] - ts[0] + 1e-10)
                feats[prefix + "stetson_j"] = compute_stetson_j(f_flux[si], f_err[si])
            else:
                feats[prefix + "rate_max"] = 0
                feats[prefix + "rate_mean"] = 0
                feats[prefix + "peak_phase"] = 0.5
                feats[prefix + "stetson_j"] = 0
        else:
            for sfx in ["mean", "std", "max", "min", "range", "skew", "kurtosis",
                         "snr", "wmean", "mad", "p10", "p90", "positive_frac",
                         "rate_max", "rate_mean", "peak_phase", "stetson_j"]:
                feats[prefix + sfx] = 0
            band_means[filt] = 0
            band_peak_flux[filt] = 0

    # --- Color features ---
    color_pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("z", "y"),
                   ("g", "i"), ("u", "r"), ("r", "z")]
    for b1, b2 in color_pairs:
        feats[f"color_{b1}_{b2}"] = band_means[b1] - band_means[b2]
        feats[f"color_peak_{b1}_{b2}"] = band_peak_flux[b1] - band_peak_flux[b2]

    # Color variability
    feats["color_g_r_range"] = (feats.get("f_g_range", 0) - feats.get("f_r_range", 0))
    feats["color_r_i_range"] = (feats.get("f_r_range", 0) - feats.get("f_i_range", 0))

    # --- Cross-filter timing ---
    peak_times = {}
    for filt in FILTERS:
        mask = filters == filt
        f_flux = flux[mask]
        f_time = time[mask]
        if len(f_flux) > 0:
            peak_times[filt] = f_time[np.argmax(f_flux)]
        else:
            peak_times[filt] = np.nan

    for b1, b2 in [("g", "r"), ("r", "i"), ("i", "z")]:
        if not np.isnan(peak_times.get(b1, np.nan)) and not np.isnan(peak_times.get(b2, np.nan)):
            feats[f"peak_delay_{b1}_{b2}"] = peak_times[b1] - peak_times[b2]
        else:
            feats[f"peak_delay_{b1}_{b2}"] = 0

    return feats


def build_features(lc_df, meta_df, desc="Features"):
    grouped = lc_df.groupby("object_id")
    records = []
    for obj_id, group in tqdm(grouped, desc=desc):
        feats = extract_features_for_object(group)
        feats["object_id"] = obj_id
        records.append(feats)
    feat_df = pd.DataFrame(records)
    merged = meta_df.merge(feat_df, on="object_id", how="left")

    merged["Z"] = pd.to_numeric(merged["Z"], errors="coerce")
    merged["Z_err"] = pd.to_numeric(merged["Z_err"], errors="coerce")
    merged["EBV"] = pd.to_numeric(merged["EBV"], errors="coerce")

    # Interaction features
    merged["flux_mean_over_z"] = merged["flux_mean"] / (merged["Z"] + 1e-10)
    merged["flux_range_over_z"] = merged["flux_range"] / (merged["Z"] + 1e-10)
    merged["amplitude_over_z"] = merged["amplitude_snr"] / (merged["Z"] + 1e-10)
    merged["peak_flux_over_z"] = merged["peak_flux"] / (merged["Z"] + 1e-10)
    merged["Z_times_EBV"] = merged["Z"] * merged["EBV"]
    merged["log_flux_range"] = np.log1p(np.abs(merged["flux_range"]))
    merged["log_peak_flux"] = np.log1p(np.abs(merged["peak_flux"]))

    # Has Z_err (train doesn't, test does) - useful binary indicator
    merged["has_z_err"] = (~merged["Z_err"].isna()).astype(int)

    return merged


# =============================================================================
# 3. Model Training
# =============================================================================
def get_feature_columns(df):
    exclude = {"object_id", "SpecType", "English Translation", "split", "target"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


def train_and_predict(train_df, test_df):
    feature_cols = get_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features")

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["target"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos = neg / max(pos, 1)
    print(f"Class ratio: {neg}:{pos}, scale_pos_weight={scale_pos:.1f}")

    # --- Model configs ---
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 3,
        "scale_pos_weight": scale_pos,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 3,
        "scale_pos_weight": scale_pos,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbosity": 0,
        "nthread": -1,
        "random_state": 42,
    }

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_et = np.zeros(len(y))
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_et = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_train)
        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=3000,
            valid_sets=[lgb_val_ds],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / n_folds

        # XGBoost
        xgb_tr = xgb.DMatrix(X_tr, label=y_tr)
        xgb_vl = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            xgb_params, xgb_tr, num_boost_round=3000,
            evals=[(xgb_vl, "val")], early_stopping_rounds=150, verbose_eval=False,
        )
        oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
        test_xgb += xgb_model.predict(xgb.DMatrix(X_test)) / n_folds

        # Extra Trees
        et = ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        et.fit(X_tr, y_tr)
        oof_et[val_idx] = et.predict_proba(X_val)[:, 1]
        test_et += et.predict_proba(X_test)[:, 1] / n_folds

        # Fold score
        oof_ens = 0.4 * oof_lgb[val_idx] + 0.4 * oof_xgb[val_idx] + 0.2 * oof_et[val_idx]
        best_f1 = max(
            f1_score(y_val, (oof_ens >= t).astype(int))
            for t in np.arange(0.05, 0.9, 0.01)
        )
        print(f"  Fold {fold+1}: Best F1={best_f1:.4f}")

    # --- Find best ensemble weights on OOF ---
    print("\nSearching best ensemble weights...")
    best_f1_global = 0
    best_w = (0.4, 0.4, 0.2)
    best_thr = 0.5

    for w_lgb in np.arange(0.2, 0.7, 0.1):
        for w_xgb in np.arange(0.2, 0.7, 0.1):
            w_et = 1.0 - w_lgb - w_xgb
            if w_et < 0.05 or w_et > 0.5:
                continue
            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_et * oof_et
            for thr in np.arange(0.05, 0.8, 0.005):
                f1 = f1_score(y, (oof_ens >= thr).astype(int))
                if f1 > best_f1_global:
                    best_f1_global = f1
                    best_w = (w_lgb, w_xgb, w_et)
                    best_thr = thr

    print(f"Best weights: LGB={best_w[0]:.1f}, XGB={best_w[1]:.1f}, ET={best_w[2]:.1f}")
    print(f"Best OOF F1: {best_f1_global:.4f} at threshold={best_thr:.3f}")

    oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_et
    print(classification_report(y, (oof_final >= best_thr).astype(int), target_names=["Non-TDE", "TDE"]))

    # Final test prediction
    test_preds = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_et

    # Feature importance (LightGBM on full data)
    lgb_full = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": 1000})
    lgb_full.fit(X, y)
    imp = pd.DataFrame({"feature": feature_cols, "importance": lgb_full.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("\nTop 20 features:")
    print(imp.head(20).to_string(index=False))

    return test_preds, best_thr


# =============================================================================
# 4. Main
# =============================================================================
def main():
    print("=" * 60)
    print("MALLORN v2 - Enhanced Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading metadata...")
    train_log, test_log = load_metadata()

    print("\n[2/4] Loading light curves...")
    train_lc, test_lc = load_all_lightcurves()

    print("\n[3/4] Feature engineering...")
    train_feat = build_features(train_lc, train_log, desc="Train features")
    test_feat = build_features(test_lc, test_log, desc="Test features")

    print("\n[4/4] Training models...")
    test_probs, threshold = train_and_predict(train_feat, test_feat)

    submission = pd.DataFrame({
        "object_id": test_log["object_id"],
        "prediction": (test_probs >= threshold).astype(int),
    })

    print(f"\nSubmission: {len(submission)} rows, Predicted TDEs: {submission['prediction'].sum()}")
    submission.to_csv("submission.csv", index=False)
    print("Saved to submission.csv")

    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    print("Format verified OK!")


if __name__ == "__main__":
    main()
