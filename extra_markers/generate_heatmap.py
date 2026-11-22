
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================
# LOAD DATA
# =====================================

CSV_FILE = "rssi_measurements.csv"

df = pd.read_csv(CSV_FILE)

print("Available columns:", df.columns.tolist())

# =====================================
# VALIDATE DATA
# =====================================

required_columns = ["rssi", "sequence"]

for col in required_columns:

    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )

df["rssi"] = pd.to_numeric(
    df["rssi"],
    errors="coerce"
)

df["sequence"] = pd.to_numeric(
    df["sequence"],
    errors="coerce"
)

df = df.dropna(
    subset=["rssi", "sequence"]
)

if df.empty:
    raise ValueError("No valid RSSI measurements found.")

# =====================================
# STATISTICS
# =====================================

print("\n========== RSSI SUMMARY ==========")

print("Total measurements:", len(df))

print("Average RSSI:", round(df["rssi"].mean(), 2), "dBm")

print("Maximum RSSI:", df["rssi"].max(), "dBm")

print("Minimum RSSI:", df["rssi"].min(), "dBm")

# =====================================
# RSSI MEASUREMENT PROFILE
# =====================================

plt.figure(figsize=(12, 6))

plt.scatter(
    df["sequence"],
    df["rssi"],
    c=df["rssi"],
    cmap="RdYlGn",
    s=40
)

plt.plot(
    df["sequence"],
    df["rssi"],
    alpha=0.4
)

plt.colorbar(
    label="RSSI (dBm)"
)

plt.xlabel("Measurement Sequence")

plt.ylabel("RSSI (dBm)")

plt.title(
    "COVERAGE-AI - Wireless Signal Strength Profile"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "rssi_profile.png",
    dpi=300
)

plt.show()

# =====================================
# RSSI DISTRIBUTION
# =====================================

plt.figure(figsize=(9, 5))

plt.hist(
    df["rssi"],
    bins=15,
    edgecolor="black"
)

plt.xlabel("RSSI (dBm)")

plt.ylabel("Number of Measurements")

plt.title("RSSI Distribution")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "rssi_distribution.png",
    dpi=300
)

plt.show()

print("\nGraphs generated successfully!")
