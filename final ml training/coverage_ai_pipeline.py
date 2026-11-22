"""
COVERAGE-AI: COMPLETE OFFLINE PIPELINE
-------------------------------------
Runs, in order:
1. RSSI profile and distribution plots
2. Timestamp association of RSSI with camera/ArUco positions
3. Spatial measured-RSSI heatmap
4. Random Forest training and spatial cross-validation
5. Full-arena AI prediction grid and AI heatmap
6. Additional-AP candidate search and before/after comparison

Place this file inside: Coverage AI/final ml training/
Expected inputs:
  extra_markers/rssi_measurements.csv
  cam test/coverage_ai_tracking.csv

Run:
  python "final ml training/coverage_ai_pipeline.py"

All generated CSV, model, and PNG files are saved beside this script.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.interpolate import griddata
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

# This script is stored in "final ml training".
# Input CSVs are stored in the sibling folders "extra_markers" and "cam test".
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RSSI_FILE = PROJECT_DIR / "extra_markers" / "rssi_measurements.csv"
TRACKING_FILE = PROJECT_DIR / "cam test" / "coverage_ai_tracking.csv"

# Keep generated outputs beside this master script.
FUSED_FILE = BASE_DIR / "coverage_ai_fused.csv"
MEASURED_HEATMAP_FILE = BASE_DIR / "coverage_ai_heatmap.png"

RSSI_PROFILE_FILE = BASE_DIR / "rssi_profile.png"
RSSI_DISTRIBUTION_FILE = BASE_DIR / "rssi_distribution.png"

MODEL_FILE = BASE_DIR / "coverage_ai_model.pkl"
AI_HEATMAP_FILE = BASE_DIR / "coverage_ai_ai_heatmap.png"
PREDICTION_FILE = BASE_DIR / "coverage_ai_predictions.csv"

AP_RESULT_FILE = BASE_DIR / "optimized_ap_placement.csv"
AP_CANDIDATES_FILE = BASE_DIR / "ap_candidate_evaluation.csv"
AP_HEATMAP_FILE = BASE_DIR / "coverage_ai_optimized_ap.png"

ARENA_SIZE = 1.5
GRID_RESOLUTION = 200
MAX_TIME_DIFFERENCE_MS = 750

WEAK_RSSI_THRESHOLD = -67.0
RANDOM_STATE = 42

# Existing Airtel router location (corrected)
EXISTING_AP_X = 0.75
EXISTING_AP_Y = 1.40

# Additional AP optimizer settings
TARGET_RSSI = -67.0
EDGE_MARGIN = 0.10
MIN_DISTANCE_FROM_EXISTING_AP = 0.30
MIN_MODEL_DISTANCE = 0.10
CANDIDATE_STEP = 0.05

# The simulated second AP is assumed to have similar effective
# transmit power and antenna characteristics to the existing AP.
# This is an assumption, not a measured fact.


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def numeric_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


# ============================================================
# 3. RSSI PROFILE + DISTRIBUTION PNGS
# ============================================================

def generate_rssi_plots():
    section("STEP 1 - RSSI PROFILE AND DISTRIBUTION")

    require_file(RSSI_FILE, "RSSI input CSV")
    df = pd.read_csv(RSSI_FILE)

    print("Available RSSI columns:", df.columns.tolist())

    df = numeric_columns(df, ["rssi", "sequence"])
    df = df.dropna(subset=["rssi", "sequence"]).copy()

    if df.empty:
        raise ValueError("No valid RSSI readings in rssi_measurements.csv")

    print("Measurements:", len(df))
    print("Average RSSI:", round(df["rssi"].mean(), 2), "dBm")
    print("Maximum RSSI:", round(df["rssi"].max(), 2), "dBm")
    print("Minimum RSSI:", round(df["rssi"].min(), 2), "dBm")

    ordered = df.sort_values("sequence")

    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        ordered["sequence"],
        ordered["rssi"],
        c=ordered["rssi"],
        cmap="RdYlGn",
        s=35
    )
    plt.plot(
        ordered["sequence"],
        ordered["rssi"],
        alpha=0.35
    )
    plt.colorbar(scatter, label="RSSI (dBm)")
    plt.xlabel("Measurement Sequence")
    plt.ylabel("RSSI (dBm)")
    plt.title("COVERAGE-AI - Wireless Signal Strength Profile")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RSSI_PROFILE_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(df["rssi"], bins=15, edgecolor="black")
    plt.xlabel("RSSI (dBm)")
    plt.ylabel("Number of Measurements")
    plt.title("COVERAGE-AI - RSSI Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RSSI_DISTRIBUTION_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", RSSI_PROFILE_FILE.name)
    print("Saved:", RSSI_DISTRIBUTION_FILE.name)


# ============================================================
# 4. TIMESTAMP ASSOCIATION + MEASURED HEATMAP
# ============================================================

def associate_rssi_and_tracking():
    section("STEP 2 - RSSI / ARUCO TIMESTAMP ASSOCIATION")

    require_file(RSSI_FILE, "RSSI input CSV")
    require_file(TRACKING_FILE, "Camera tracking CSV")

    print("RSSI file:", RSSI_FILE)
    print("Tracking file:", TRACKING_FILE)

    rssi_df = pd.read_csv(RSSI_FILE)
    tracking_df = pd.read_csv(TRACKING_FILE)

    print("RSSI records:", len(rssi_df))
    print("Tracking records:", len(tracking_df))

    required_rssi = ["timestamp", "device_id", "rssi", "sequence"]
    required_tracking = ["timestamp", "camera", "smooth_x", "smooth_y"]

    for col in required_rssi:
        if col not in rssi_df.columns:
            raise ValueError(f"Missing RSSI column: {col}")

    for col in required_tracking:
        if col not in tracking_df.columns:
            raise ValueError(f"Missing tracking column: {col}")

    # Parse timestamps consistently. utc=True prevents mixed timezone
    # formats from causing merge_asof comparison errors.
    rssi_df["timestamp"] = pd.to_datetime(
        rssi_df["timestamp"], errors="coerce", utc=True
    )
    tracking_df["timestamp"] = pd.to_datetime(
        tracking_df["timestamp"], errors="coerce", utc=True
    )

    rssi_df = numeric_columns(rssi_df, ["rssi", "sequence"])
    tracking_df = numeric_columns(
        tracking_df, ["smooth_x", "smooth_y"]
    )

    rssi_df = rssi_df.dropna(
        subset=["timestamp", "rssi", "sequence"]
    ).copy()

    tracking_df = tracking_df.dropna(
        subset=["timestamp", "smooth_x", "smooth_y"]
    ).copy()

    tracking_df = tracking_df.rename(
        columns={"smooth_x": "x", "smooth_y": "y"}
    )

    tracking_df = tracking_df[
        tracking_df["x"].between(0, ARENA_SIZE)
        & tracking_df["y"].between(0, ARENA_SIZE)
    ].copy()

    tracking_match = tracking_df[
        ["timestamp", "camera", "x", "y"]
    ].rename(columns={"timestamp": "position_timestamp"})

    if rssi_df.empty:
        raise ValueError("No valid RSSI records after cleaning.")
    if tracking_match.empty:
        raise ValueError("No valid camera positions after cleaning.")

    # merge_asof requires both dataframes sorted by their join key.
    rssi_df = rssi_df.sort_values("timestamp").reset_index(drop=True)
    tracking_match = tracking_match.sort_values(
        "position_timestamp"
    ).reset_index(drop=True)

    print("\nMatching RSSI measurements to nearest camera position...")

    fused = pd.merge_asof(
        rssi_df,
        tracking_match,
        left_on="timestamp",
        right_on="position_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=MAX_TIME_DIFFERENCE_MS)
    )

    fused["time_difference_ms"] = (
        fused["timestamp"] - fused["position_timestamp"]
    ).abs().dt.total_seconds() * 1000

    matched = fused.dropna(
        subset=["x", "y", "position_timestamp", "camera"]
    ).copy()

    matched.to_csv(FUSED_FILE, index=False)

    print("\nTIMESTAMP MATCHING RESULTS")
    print("RSSI records:", len(rssi_df))
    print("Camera positions:", len(tracking_match))
    print("Matched records:", len(matched))
    print("Unmatched records:", len(rssi_df) - len(matched))

    if not matched.empty:
        print(
            "Mean timestamp difference:",
            round(matched["time_difference_ms"].mean(), 2),
            "ms"
        )
        print(
            "Maximum timestamp difference:",
            round(matched["time_difference_ms"].max(), 2),
            "ms"
        )

    print("Fused CSV saved:", FUSED_FILE)

    if len(matched) < 3:
        raise ValueError(
            "Fewer than 3 matched RSSI/position records. "
            "Check camera tracking and timestamps."
        )

    spatial = (
        matched.groupby(["x", "y"], as_index=False)["rssi"]
        .mean()
    )

    if len(spatial) < 3:
        raise ValueError(
            "Need at least 3 distinct matched positions for measured heatmap."
        )

    points = spatial[["x", "y"]].to_numpy()
    values = spatial["rssi"].to_numpy()

    gx, gy = np.mgrid[
        0:ARENA_SIZE:200j,
        0:ARENA_SIZE:200j
    ]

    # Linear interpolation may leave areas outside the measured convex hull
    # empty. Fill those areas with nearest-neighbour estimates for a complete
    # visualization, while remembering these are extrapolated estimates.
    grid_rssi = griddata(points, values, (gx, gy), method="linear")
    nearest_grid = griddata(points, values, (gx, gy), method="nearest")
    grid_rssi = np.where(np.isnan(grid_rssi), nearest_grid, grid_rssi)

    plt.figure(figsize=(10, 8))
    heatmap = plt.contourf(
        gx, gy, grid_rssi,
        levels=25,
        cmap="RdYlGn",
        extend="both"
    )
    plt.scatter(
        points[:, 0], points[:, 1],
        c=values,
        cmap="RdYlGn",
        edgecolors="black",
        s=30,
        zorder=3
    )
    plt.colorbar(heatmap, label="Measured RSSI (dBm)")
    plt.xlim(0, ARENA_SIZE)
    plt.ylim(0, ARENA_SIZE)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("COVERAGE-AI - Spatial Measured RSSI Heatmap")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(MEASURED_HEATMAP_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print("Measured heatmap saved:", MEASURED_HEATMAP_FILE)


# ============================================================
# 5. RANDOM FOREST TRAINING + FULL ARENA AI HEATMAP
# ============================================================

def train_model_and_generate_ai_heatmap():
    section("STEP 3 - RANDOM FOREST TRAINING AND AI HEATMAP")

    require_file(FUSED_FILE, "Fused training dataset")
    df = pd.read_csv(FUSED_FILE)

    df = numeric_columns(df, ["x", "y", "rssi"])
    df = df.dropna(subset=["x", "y", "rssi"]).copy()

    df = df[
        df["x"].between(0, ARENA_SIZE)
        & df["y"].between(0, ARENA_SIZE)
        & df["rssi"].between(-100, 0)
    ].copy()

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} valid fused records. "
            "Collect at least 20 observations."
        )

    unique_positions = df[["x", "y"]].drop_duplicates()
    if len(unique_positions) < 10:
        raise ValueError(
            "Need at least 10 distinct spatial positions for training."
        )

    print("Training records:", len(df))
    print("Unique positions:", len(unique_positions))

    X = df[["x", "y"]].copy()
    y = df["rssi"].copy()

    groups = (
        np.floor(df["x"] / 0.2).astype(int).astype(str)
        + "_"
        + np.floor(df["y"] / 0.2).astype(int).astype(str)
    )

    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(
            "Not enough spatial groups for GroupKFold validation."
        )

    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    oof_predictions = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        fold_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_predictions[test_idx] = fold_model.predict(X.iloc[test_idx])
        print(f"Spatial validation fold {fold}/{n_splits} complete")

    mae = mean_absolute_error(y, oof_predictions)
    rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    r2 = r2_score(y, oof_predictions)

    print("\nMODEL PERFORMANCE")
    print(f"MAE:  {mae:.3f} dB")
    print(f"RMSE: {rmse:.3f} dB")
    print(f"R2:   {r2:.3f}")

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)

    model_package = {
        "model": model,
        "features": ["x", "y"],
        "target": "rssi",
        "arena_size": ARENA_SIZE,
        "weak_threshold": WEAK_RSSI_THRESHOLD,
        "training_samples": len(df),
        "unique_positions": len(unique_positions),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }
    }
    joblib.dump(model_package, MODEL_FILE)
    print("Trained model saved:", MODEL_FILE)

    gx, gy = np.meshgrid(
        np.linspace(0, ARENA_SIZE, GRID_RESOLUTION),
        np.linspace(0, ARENA_SIZE, GRID_RESOLUTION)
    )
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    predicted = model.predict(
        pd.DataFrame(grid_points, columns=["x", "y"])
    ).reshape(gx.shape)

    weak_mask = predicted < WEAK_RSSI_THRESHOLD
    weak_percentage = float(np.mean(weak_mask) * 100)

    prediction_df = pd.DataFrame({
        "x": gx.ravel(),
        "y": gy.ravel(),
        "predicted_rssi": predicted.ravel()
    })
    prediction_df["coverage_status"] = np.where(
        prediction_df["predicted_rssi"] < WEAK_RSSI_THRESHOLD,
        "WEAK",
        "ACCEPTABLE"
    )
    prediction_df.to_csv(PREDICTION_FILE, index=False)

    print("\nAI COVERAGE ANALYSIS")
    print("Grid:", GRID_RESOLUTION, "x", GRID_RESOLUTION)
    print("Weak threshold:", WEAK_RSSI_THRESHOLD, "dBm")
    print("Weak predicted area:", round(weak_percentage, 2), "%")
    print(
        "Predicted RSSI range:",
        round(float(predicted.min()), 2),
        "to",
        round(float(predicted.max()), 2),
        "dBm"
    )
    print("Prediction grid saved:", PREDICTION_FILE)

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(
        predicted,
        extent=[0, ARENA_SIZE, 0, ARENA_SIZE],
        origin="lower",
        cmap="RdYlGn",
        interpolation="bilinear",
        aspect="equal",
        vmin=-80,
        vmax=-30
    )

    if predicted.min() < WEAK_RSSI_THRESHOLD < predicted.max():
        ax.contour(
            gx, gy, predicted,
            levels=[WEAK_RSSI_THRESHOLD],
            colors="black",
            linewidths=1.5
        )

    ax.scatter(
        df["x"], df["y"],
        color="black",
        s=13,
        label="Actual RSSI measurements",
        zorder=3
    )
    ax.scatter(
        [EXISTING_AP_X], [EXISTING_AP_Y],
        marker="D",
        color="blue",
        edgecolors="white",
        s=110,
        label="Existing Airtel AP",
        zorder=5
    )
    fig.colorbar(image, ax=ax, label="AI Predicted RSSI (dBm)")
    ax.set_xlim(0, ARENA_SIZE)
    ax.set_ylim(0, ARENA_SIZE)
    ax.set_xlabel("X Position (metres)")
    ax.set_ylabel("Y Position (metres)")
    ax.set_title(
        "COVERAGE-AI | FULL ARENA WIFI COVERAGE\n"
        f"Random Forest | Weak threshold: {WEAK_RSSI_THRESHOLD} dBm"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(AI_HEATMAP_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print("AI heatmap saved:", AI_HEATMAP_FILE)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "weak_percentage": weak_percentage
    }


# ============================================================
# 6. ADDITIONAL AP PLACEMENT OPTIMIZATION
# ============================================================

def optimize_additional_ap():
    section("STEP 4 - ADDITIONAL AP PLACEMENT OPTIMIZATION")

    require_file(PREDICTION_FILE, "AI prediction grid")
    require_file(FUSED_FILE, "Fused measurements")

    pred_df = pd.read_csv(PREDICTION_FILE)
    meas_df = pd.read_csv(FUSED_FILE)

    pred_df = numeric_columns(
        pred_df, ["x", "y", "predicted_rssi"]
    ).dropna(subset=["x", "y", "predicted_rssi"])

    meas_df = numeric_columns(
        meas_df, ["x", "y", "rssi"]
    ).dropna(subset=["x", "y", "rssi"])

    meas_df = meas_df[meas_df["rssi"].between(-100, 0)].copy()

    if len(meas_df) < 10:
        raise ValueError(
            "Not enough fused measurements to calibrate AP propagation."
        )

    x_values = np.sort(pred_df["x"].unique())
    y_values = np.sort(pred_df["y"].unique())

    existing_grid = (
        pred_df.pivot(
            index="y",
            columns="x",
            values="predicted_rssi"
        )
        .reindex(index=y_values, columns=x_values)
        .to_numpy()
    )

    if not np.isfinite(existing_grid).all():
        raise ValueError("AI prediction grid contains missing values.")

    grid_x, grid_y = np.meshgrid(x_values, y_values)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    existing_flat = existing_grid.ravel()

    print("\nExisting Airtel AP:")
    print(f"X = {EXISTING_AP_X:.2f} m")
    print(f"Y = {EXISTING_AP_Y:.2f} m")

    # Fit log-distance path-loss model from the fused observations.
    distances = np.sqrt(
        (meas_df["x"].to_numpy() - EXISTING_AP_X) ** 2
        + (meas_df["y"].to_numpy() - EXISTING_AP_Y) ** 2
    )
    distances = np.maximum(distances, MIN_MODEL_DISTANCE)
    log_distances = np.log10(distances)
    observed_rssi = meas_df["rssi"].to_numpy()

    def residuals(params):
        intercept, slope = params
        return intercept + slope * log_distances - observed_rssi

    fit = least_squares(
        residuals,
        x0=[float(np.median(observed_rssi)), -20.0],
        bounds=([-150.0, -60.0], [50.0, -10.0]),
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=5000
    )

    intercept = float(fit.x[0])
    slope = float(fit.x[1])
    path_loss_exponent = -slope / 10.0
    fitted_rssi = intercept + slope * log_distances
    calibration_mae = float(
        np.mean(np.abs(observed_rssi - fitted_rssi))
    )

    print("\nPATH-LOSS CALIBRATION")
    print("Intercept:", round(intercept, 2), "dBm")
    print("Path-loss exponent:", round(path_loss_exponent, 3))
    print("Calibration MAE:", round(calibration_mae, 2), "dB")

    if calibration_mae > 8:
        warnings.warn(
            "Path-loss calibration MAE exceeds 8 dB. "
            "AP placement output is exploratory and should not be "
            "treated as a reliable physical recommendation."
        )

    before_coverage = float(
        np.mean(existing_flat >= TARGET_RSSI) * 100
    )
    print("\nCoverage before additional AP:", round(before_coverage, 2), "%")

    candidate_coordinates = np.arange(
        EDGE_MARGIN,
        ARENA_SIZE - EDGE_MARGIN + 0.0001,
        CANDIDATE_STEP
    )

    candidate_results = []

    for candidate_x in candidate_coordinates:
        for candidate_y in candidate_coordinates:

            separation = np.hypot(
                candidate_x - EXISTING_AP_X,
                candidate_y - EXISTING_AP_Y
            )

            if separation < MIN_DISTANCE_FROM_EXISTING_AP:
                continue

            candidate_distances = np.hypot(
                points[:, 0] - candidate_x,
                points[:, 1] - candidate_y
            )
            candidate_distances = np.maximum(
                candidate_distances, MIN_MODEL_DISTANCE
            )

            new_ap_rssi = (
                intercept + slope * np.log10(candidate_distances)
            )

            # Approximation: clients can use whichever AP is stronger.
            combined_rssi = np.maximum(existing_flat, new_ap_rssi)

            coverage_after = float(
                np.mean(combined_rssi >= TARGET_RSSI) * 100
            )

            candidate_results.append({
                "ap_x": float(candidate_x),
                "ap_y": float(candidate_y),
                "distance_from_existing_ap": float(separation),
                "coverage_before_percent": before_coverage,
                "coverage_after_percent": coverage_after,
                "coverage_improvement_percentage_points":
                    coverage_after - before_coverage,
                "average_rssi_after_dbm": float(np.mean(combined_rssi)),
                "weak_grid_points_after": int(
                    np.sum(combined_rssi < TARGET_RSSI)
                )
            })

    results_df = pd.DataFrame(candidate_results)
    if results_df.empty:
        raise ValueError("No valid additional AP candidate locations.")

    results_df = results_df.sort_values(
        ["coverage_after_percent", "average_rssi_after_dbm"],
        ascending=False
    ).reset_index(drop=True)

    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))
    results_df.to_csv(AP_CANDIDATES_FILE, index=False)

    best = results_df.iloc[0]
    best_x = float(best["ap_x"])
    best_y = float(best["ap_y"])

    best_distances = np.hypot(
        points[:, 0] - best_x,
        points[:, 1] - best_y
    )
    best_distances = np.maximum(best_distances, MIN_MODEL_DISTANCE)

    best_new_ap_rssi = (
        intercept + slope * np.log10(best_distances)
    )
    combined_grid = np.maximum(
        existing_flat, best_new_ap_rssi
    ).reshape(existing_grid.shape)

    pd.DataFrame([{
        "existing_ap_x": EXISTING_AP_X,
        "existing_ap_y": EXISTING_AP_Y,
        "recommended_ap_x": best_x,
        "recommended_ap_y": best_y,
        "target_rssi_dbm": TARGET_RSSI,
        "coverage_before_percent": before_coverage,
        "coverage_after_percent": float(best["coverage_after_percent"]),
        "coverage_improvement_percentage_points":
            float(best["coverage_improvement_percentage_points"]),
        "path_loss_exponent": path_loss_exponent,
        "calibration_mae_db": calibration_mae
    }]).to_csv(AP_RESULT_FILE, index=False)

    # Before/after image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    extent = [0, ARENA_SIZE, 0, ARENA_SIZE]

    im1 = axes[0].imshow(
        existing_grid,
        extent=extent,
        origin="lower",
        cmap="RdYlGn",
        vmin=-80,
        vmax=-30,
        interpolation="nearest",
        aspect="equal"
    )
    axes[0].scatter(
        EXISTING_AP_X, EXISTING_AP_Y,
        marker="D", s=150, color="blue",
        edgecolors="white", label="Existing Airtel AP",
        zorder=5
    )
    axes[0].set_title(
        f"BEFORE ADDITIONAL AP\nCoverage: {before_coverage:.1f}%"
    )

    im2 = axes[1].imshow(
        combined_grid,
        extent=extent,
        origin="lower",
        cmap="RdYlGn",
        vmin=-80,
        vmax=-30,
        interpolation="nearest",
        aspect="equal"
    )
    axes[1].scatter(
        EXISTING_AP_X, EXISTING_AP_Y,
        marker="D", s=150, color="blue",
        edgecolors="white", label="Existing Airtel AP",
        zorder=5
    )
    axes[1].scatter(
        best_x, best_y,
        marker="*", s=500, color="magenta",
        edgecolors="white", linewidths=1.5,
        label="Recommended Additional AP",
        zorder=6
    )
    axes[1].set_title(
        f"AFTER ADDITIONAL AP\n"
        f"Coverage: {best['coverage_after_percent']:.1f}%"
    )

    for ax in axes:
        ax.set_xlim(0, ARENA_SIZE)
        ax.set_ylim(0, ARENA_SIZE)
        ax.set_xlabel("X Position (metres)")
        ax.set_ylabel("Y Position (metres)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)

    fig.colorbar(
        im2, ax=axes,
        label="Predicted RSSI (dBm)",
        shrink=0.85
    )
    fig.suptitle(
        "COVERAGE-AI | OPTIMIZED ACCESS POINT PLACEMENT",
        fontsize=15,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(AP_HEATMAP_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nAP PLACEMENT RESULT")
    print(f"Existing AP: ({EXISTING_AP_X:.2f}, {EXISTING_AP_Y:.2f}) m")
    print(f"Candidate AP: ({best_x:.2f}, {best_y:.2f}) m")
    print(f"Coverage before: {before_coverage:.2f}%")
    print(f"Coverage after: {best['coverage_after_percent']:.2f}%")
    print(
        "Improvement:",
        f"{best['coverage_improvement_percentage_points']:.2f}",
        "percentage points"
    )
    print("Candidate evaluation CSV:", AP_CANDIDATES_FILE)
    print("Recommendation CSV:", AP_RESULT_FILE)
    print("AP comparison PNG:", AP_HEATMAP_FILE)

    if calibration_mae > 8:
        print(
            "\nCAUTION: Calibration error is high. "
            "Treat this AP location as an exploratory candidate only."
        )


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    section("COVERAGE-AI COMPLETE PIPELINE")
    print("Working folder:", BASE_DIR)
    print("Existing Airtel AP:", (EXISTING_AP_X, EXISTING_AP_Y))

    generate_rssi_plots()
    associate_rssi_and_tracking()
    train_model_and_generate_ai_heatmap()
    optimize_additional_ap()

    section("PIPELINE COMPLETED - GENERATED PNG IMAGES")
    for image_path in [
        RSSI_PROFILE_FILE,
        RSSI_DISTRIBUTION_FILE,
        MEASURED_HEATMAP_FILE,
        AI_HEATMAP_FILE,
        AP_HEATMAP_FILE
    ]:
        print(("OK  " if image_path.exists() else "MISSING  ") + str(image_path))

    print("\nOther outputs:")
    for output_path in [
        FUSED_FILE,
        MODEL_FILE,
        PREDICTION_FILE,
        AP_RESULT_FILE,
        AP_CANDIDATES_FILE
    ]:
        print(("OK  " if output_path.exists() else "MISSING  ") + str(output_path))


if __name__ == "__main__":
    main()
