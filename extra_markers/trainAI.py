
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "coverage_ai_fused.csv"

MODEL_FILE = BASE_DIR / "coverage_ai_model.pkl"

HEATMAP_FILE = BASE_DIR / "coverage_ai_ai_heatmap.png"

PREDICTION_FILE = BASE_DIR / "coverage_ai_predictions.csv"

ARENA_SIZE = 1.5

GRID_RESOLUTION = 200

WEAK_RSSI_THRESHOLD = -67

RANDOM_STATE = 42

# =====================================================
# 2. START
# =====================================================

print("\n")
print("=" * 55)
print("             COVERAGE-AI")
print("       MACHINE LEARNING ENGINE")
print("=" * 55)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {DATA_FILE}"
    )

# =====================================================
# 3. LOAD DATASET
# =====================================================

print("\n[1/10] Loading dataset...")

df = pd.read_csv(DATA_FILE)

required_columns = ["x", "y", "rssi"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column missing: {col}"
        )

print("Total records:", len(df))

# =====================================================
# 4. DATA CLEANING
# =====================================================

print("\n[2/10] Cleaning measurements...")

for col in required_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=required_columns
).copy()

# Remove invalid coordinates
df = df[
    (df["x"] >= 0) &
    (df["x"] <= ARENA_SIZE) &
    (df["y"] >= 0) &
    (df["y"] <= ARENA_SIZE)
].copy()

# Remove invalid RSSI readings
df = df[
    (df["rssi"] >= -100) &
    (df["rssi"] <= 0)
].copy()

if len(df) < 20:
    raise ValueError(
        "Insufficient measurements. "
        "Collect at least 20 valid observations."
    )

unique_positions = df[
    ["x", "y"]
].drop_duplicates()

print("Valid records:", len(df))
print("Unique positions:", len(unique_positions))

print("\nRSSI statistics:")
print(df["rssi"].describe())

if len(unique_positions) < 10:
    raise ValueError(
        "Collect measurements at more distinct positions."
    )

# =====================================================
# 5. PREPARE TRAINING DATA
# =====================================================

print("\n[3/10] Preparing training data...")

X = df[["x", "y"]].copy()

y = df["rssi"].copy()

# Spatial grouping for validation
# Nearby positions are kept in the same validation group.

groups = (
    np.floor(df["x"] / 0.2).astype(int).astype(str)
    + "_"
    + np.floor(df["y"] / 0.2).astype(int).astype(str)
)

n_groups = groups.nunique()

print("Spatial groups:", n_groups)

if n_groups < 2:
    raise ValueError(
        "Not enough spatial groups for model validation."
    )

# =====================================================
# 6. SPATIAL MODEL VALIDATION
# =====================================================

print("\n[4/10] Evaluating Random Forest model...")

n_splits = min(5, n_groups)

cv = GroupKFold(
    n_splits=n_splits
)

oof_predictions = np.zeros(len(df))

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    fold_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    fold_model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    predictions = fold_model.predict(
        X.iloc[test_idx]
    )

    oof_predictions[test_idx] = predictions

    print(
        f"Fold {fold}/{n_splits} completed"
    )

# =====================================================
# 7. CALCULATE ACCURACY
# =====================================================

print("\n[5/10] Calculating model performance...")

mae = mean_absolute_error(
    y,
    oof_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        oof_predictions
    )
)

r2 = r2_score(
    y,
    oof_predictions
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.3f} dB")

print(f"RMSE : {rmse:.3f} dB")

print(f"R2   : {r2:.3f}")

# =====================================================
# 8. TRAIN FINAL MODEL
# =====================================================

print("\n[6/10] Training final Random Forest model...")

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X, y)

print("Model training completed!")

# =====================================================
# 9. SAVE TRAINED MODEL
# =====================================================

print("\n[7/10] Saving trained model...")

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

joblib.dump(
    model_package,
    MODEL_FILE
)

print("Model saved:", MODEL_FILE)

# =====================================================
# 10. GENERATE FULL ARENA GRID
# =====================================================

print("\n[8/10] Generating FULL ARENA predictions...")

grid_x, grid_y = np.meshgrid(
    np.linspace(
        0,
        ARENA_SIZE,
        GRID_RESOLUTION
    ),
    np.linspace(
        0,
        ARENA_SIZE,
        GRID_RESOLUTION
    )
)

grid_points = np.column_stack([
    grid_x.ravel(),
    grid_y.ravel()
])

print(
    "Grid dimensions:",
    GRID_RESOLUTION,
    "x",
    GRID_RESOLUTION
)

print(
    "Total prediction points:",
    len(grid_points)
)

# =====================================================
# 11. PREDICT RSSI ACROSS ENTIRE ARENA
# =====================================================

print("\nPredicting RSSI at every grid location...")

# No convex hull restriction.
# The model predicts throughout the complete arena.

predicted_rssi = model.predict(
    pd.DataFrame(
        grid_points,
        columns=["x", "y"]
    )
)

predicted_rssi = predicted_rssi.reshape(
    grid_x.shape
)

print("Full arena prediction completed!")

# =====================================================
# 12. IDENTIFY WEAK COVERAGE
# =====================================================

print("\n[9/10] Analysing wireless coverage...")

weak_mask = (
    predicted_rssi < WEAK_RSSI_THRESHOLD
)

weak_count = int(
    np.sum(weak_mask)
)

total_points = predicted_rssi.size

weak_percentage = (
    weak_count / total_points
) * 100

print("\n========== COVERAGE ANALYSIS ==========")

print(
    "Weak RSSI threshold:",
    WEAK_RSSI_THRESHOLD,
    "dBm"
)

print(
    "Weak grid points:",
    weak_count
)

print(
    "Acceptable grid points:",
    total_points - weak_count
)

print(
    "Weak coverage percentage:",
    round(weak_percentage, 2),
    "%"
)

print(
    "Predicted RSSI range:",
    round(float(predicted_rssi.min()), 2),
    "to",
    round(float(predicted_rssi.max()), 2),
    "dBm"
)

# =====================================================
# 13. SAVE FULL PREDICTION GRID
# =====================================================

prediction_df = pd.DataFrame({
    "x": grid_x.ravel(),
    "y": grid_y.ravel(),
    "predicted_rssi": predicted_rssi.ravel()
})

prediction_df["coverage_status"] = np.where(
    prediction_df["predicted_rssi"] <
    WEAK_RSSI_THRESHOLD,
    "WEAK",
    "ACCEPTABLE"
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print(
    "\nPrediction data saved:",
    PREDICTION_FILE
)

# =====================================================
# 14. GENERATE FULL ARENA AI HEATMAP
# =====================================================

print("\n[10/10] Generating AI heatmap...")

fig, ax = plt.subplots(
    figsize=(12, 10)
)

# Full square heatmap
heatmap = ax.imshow(
    predicted_rssi,
    extent=[
        0,
        ARENA_SIZE,
        0,
        ARENA_SIZE
    ],
    origin="lower",
    cmap="RdYlGn",
    interpolation="bilinear",
    aspect="equal",
    vmin=-80,
    vmax=-30
)

# Weak coverage boundary
if predicted_rssi.min() < WEAK_RSSI_THRESHOLD < predicted_rssi.max():

    ax.contour(
        grid_x,
        grid_y,
        predicted_rssi,
        levels=[WEAK_RSSI_THRESHOLD],
        colors="black",
        linewidths=1.5
    )

# Actual measurement positions
ax.scatter(
    df["x"],
    df["y"],
    color="black",
    s=16,
    label="Actual RSSI measurements",
    zorder=3
)

# Arena boundary
ax.plot(
    [0, ARENA_SIZE, ARENA_SIZE, 0, 0],
    [0, 0, ARENA_SIZE, ARENA_SIZE, 0],
    color="black",
    linewidth=2
)

# Colorbar
colorbar = fig.colorbar(
    heatmap,
    ax=ax
)

colorbar.set_label(
    "AI Predicted RSSI (dBm)"
)

ax.set_xlim(0, ARENA_SIZE)

ax.set_ylim(0, ARENA_SIZE)

ax.set_xlabel(
    "X Position (metres)"
)

ax.set_ylabel(
    "Y Position (metres)"
)

ax.set_title(
    "COVERAGE-AI | FULL ARENA WIFI COVERAGE\n"
    f"Random Forest | Weak threshold: "
    f"{WEAK_RSSI_THRESHOLD} dBm"
)

ax.legend(
    loc="upper right"
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    HEATMAP_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 15. FINAL REPORT
# =====================================================

print("\n")
print("=" * 55)
print("       COVERAGE-AI TRAINING COMPLETED")
print("=" * 55)

print("\nTRAINING:")
print("Measurements:", len(df))
print("Unique positions:", len(unique_positions))

print("\nMODEL PERFORMANCE:")
print("MAE:", round(mae, 3), "dB")
print("RMSE:", round(rmse, 3), "dB")
print("R2:", round(r2, 3))

print("\nCOVERAGE:")
print("Arena:", ARENA_SIZE, "x", ARENA_SIZE, "metres")
print("Grid points:", total_points)
print("Weak coverage:", round(weak_percentage, 2), "%")

print("\nGENERATED FILES:") 
print("Model:", MODEL_FILE)
print("Predictions:", PREDICTION_FILE)
print("AI Heatmap:", HEATMAP_FILE)

print("\nALL OPERATIONS COMPLETED!")
