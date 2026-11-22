
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# RSSI data inside extra_markers
RSSI_FILE = BASE_DIR / "rssi_measurements.csv"

# Tracking data inside cam test
TRACKING_FILE = (
    BASE_DIR.parent
    / "cam test"
    / "coverage_ai_tracking.csv"
)

# Output files inside extra_markers
OUTPUT_FILE = BASE_DIR / "coverage_ai_fused.csv"
HEATMAP_FILE = BASE_DIR / "coverage_ai_heatmap.png"

# Maximum allowed timestamp difference
MAX_TIME_DIFFERENCE_MS = 750

# Arena dimensions in metres
ARENA_SIZE = 1.5

# ==========================================
# 2. CHECK FILES
# ==========================================

print("\n========================================")
print("COVERAGE-AI TIMESTAMP SYNCHRONIZATION")
print("========================================")

print("\nRSSI file:", RSSI_FILE)
print("Tracking file:", TRACKING_FILE)

if not RSSI_FILE.exists():
    raise FileNotFoundError(
        f"RSSI file not found: {RSSI_FILE}"
    )

if not TRACKING_FILE.exists():
    raise FileNotFoundError(
        f"Tracking file not found: {TRACKING_FILE}"
    )

# ==========================================
# 3. LOAD CSV FILES
# ==========================================

rssi_df = pd.read_csv(RSSI_FILE)
tracking_df = pd.read_csv(TRACKING_FILE)

print("\nRSSI records:", len(rssi_df))
print("Tracking records:", len(tracking_df))

print("\nRSSI columns:")
print(rssi_df.columns.tolist())

print("\nTracking columns:")
print(tracking_df.columns.tolist())

# ==========================================
# 4. VALIDATE REQUIRED COLUMNS
# ==========================================

required_rssi = [
    "timestamp",
    "device_id",
    "rssi",
    "sequence"
]

required_tracking = [
    "timestamp",
    "camera",
    "smooth_x",
    "smooth_y"
]

for column in required_rssi:

    if column not in rssi_df.columns:
        raise ValueError(
            f"Missing RSSI column: {column}"
        )

for column in required_tracking:

    if column not in tracking_df.columns:
        raise ValueError(
            f"Missing tracking column: {column}"
        )

# ==========================================
# 5. CONVERT DATA TYPES
# ==========================================

rssi_df["timestamp"] = pd.to_datetime(
    rssi_df["timestamp"],
    errors="coerce"
)

tracking_df["timestamp"] = pd.to_datetime(
    tracking_df["timestamp"],
    errors="coerce"
)

rssi_df["rssi"] = pd.to_numeric(
    rssi_df["rssi"],
    errors="coerce"
)

rssi_df["sequence"] = pd.to_numeric(
    rssi_df["sequence"],
    errors="coerce"
)

tracking_df["smooth_x"] = pd.to_numeric(
    tracking_df["smooth_x"],
    errors="coerce"
)

tracking_df["smooth_y"] = pd.to_numeric(
    tracking_df["smooth_y"],
    errors="coerce"
)

# Remove invalid records
rssi_df = rssi_df.dropna(
    subset=["timestamp", "rssi", "sequence"]
).copy()

tracking_df = tracking_df.dropna(
    subset=["timestamp", "smooth_x", "smooth_y"]
).copy()

# ==========================================
# 6. PREPARE TRACKING COORDINATES
# ==========================================

tracking_df = tracking_df.rename(
    columns={
        "smooth_x": "x",
        "smooth_y": "y"
    }
)

# Keep coordinates inside the arena
tracking_df = tracking_df[
    (tracking_df["x"] >= 0) &
    (tracking_df["x"] <= ARENA_SIZE) &
    (tracking_df["y"] >= 0) &
    (tracking_df["y"] <= ARENA_SIZE)
].copy()

# Select required tracking columns
tracking_df = tracking_df[
    [
        "timestamp",
        "camera",
        "x",
        "y"
    ]
].copy()

# ==========================================
# 7. SORT TIMESTAMPS
# ==========================================

rssi_df = rssi_df.sort_values(
    "timestamp"
).reset_index(drop=True)

tracking_df = tracking_df.sort_values(
    "timestamp"
).reset_index(drop=True)

if rssi_df.empty:
    raise ValueError("No valid RSSI measurements found.")

if tracking_df.empty:
    raise ValueError("No valid tracking coordinates found.")

# ==========================================
# 8. MATCH RSSI WITH ARUCO POSITION
# ==========================================

print("\nMatching RSSI measurements with positions...")

# Rename tracking timestamp to preserve both timestamps
tracking_match = tracking_df.rename(
    columns={
        "timestamp": "position_timestamp"
    }
)

# Nearest timestamp matching
fused = pd.merge_asof(
    rssi_df,
    tracking_match,
    left_on="timestamp",
    right_on="position_timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=MAX_TIME_DIFFERENCE_MS
    )
)

# ==========================================
# 9. CALCULATE TIME DIFFERENCE
# ==========================================

fused["time_difference_ms"] = (
    fused["timestamp"] -
    fused["position_timestamp"]
).abs().dt.total_seconds() * 1000

# ==========================================
# 10. REMOVE UNMATCHED RECORDS
# ==========================================

matched = fused.dropna(
    subset=[
        "x",
        "y",
        "position_timestamp",
        "camera"
    ]
).copy()

unmatched = len(rssi_df) - len(matched)

print("\n========================================")
print("TIMESTAMP MATCHING RESULTS")
print("========================================")

print("Total RSSI measurements:", len(rssi_df))
print("Total camera positions:", len(tracking_df))
print("Matched measurements:", len(matched))
print("Unmatched measurements:", unmatched)

if not matched.empty:

    print(
        "Average time difference:",
        round(
            matched["time_difference_ms"].mean(),
            2
        ),
        "ms"
    )

    print(
        "Maximum time difference:",
        round(
            matched["time_difference_ms"].max(),
            2
        ),
        "ms"
    )

# ==========================================
# 11. SAVE FUSED CSV
# ==========================================

matched.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFused data saved:")
print(OUTPUT_FILE)

# ==========================================
# 12. CHECK HEATMAP REQUIREMENTS
# ==========================================

if len(matched) < 3:

    raise ValueError(
        "Not enough matched measurements "
        "to generate a spatial heatmap."
    )

# Average measurements at identical positions
spatial_data = (
    matched.groupby(
        ["x", "y"],
        as_index=False
    )["rssi"]
    .mean()
)

points = spatial_data[
    ["x", "y"]
].values

values = spatial_data[
    "rssi"
].values

if len(points) < 3:

    raise ValueError(
        "Need at least 3 distinct spatial positions "
        "for interpolation."
    )

# ==========================================
# 13. GENERATE INTERPOLATION GRID
# ==========================================

grid_x, grid_y = np.mgrid[
    0:ARENA_SIZE:200j,
    0:ARENA_SIZE:200j
]

grid_rssi = griddata(
    points,
    values,
    (grid_x, grid_y),
    method="linear"
)

# ==========================================
# 14. GENERATE RSSI HEATMAP
# ==========================================

plt.figure(figsize=(10, 8))

heatmap = plt.contourf(
    grid_x,
    grid_y,
    grid_rssi,
    levels=25,
    cmap="RdYlGn",
    extend="both"
)

# Plot actual measurement positions
plt.scatter(
    points[:, 0],
    points[:, 1],
    c=values,
    cmap="RdYlGn",
    edgecolors="black",
    s=35,
    zorder=3
)

plt.colorbar(
    heatmap,
    label="RSSI (dBm)"
)

plt.xlim(0, ARENA_SIZE)
plt.ylim(0, ARENA_SIZE)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

plt.title(
    "COVERAGE-AI - Spatial Wireless RSSI Heatmap"
)

plt.gca().set_aspect("equal")

plt.tight_layout()

# ==========================================
# 15. SAVE HEATMAP
# ==========================================

plt.savefig(
    HEATMAP_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n========================================")
print("HEATMAP GENERATION COMPLETED")
print("========================================")

print("Fused CSV:", OUTPUT_FILE)
print("Heatmap:", HEATMAP_FILE)
print("Matched samples:", len(matched))
