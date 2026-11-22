
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent

# Load cleaned dataset
df = pd.read_csv(BASE_DIR / "cleaned_wifi_data.csv")

print("========== EDA SUMMARY ==========")
print("Average RSSI:", df["rssi_mean"].mean())
print("Strongest RSSI:", df["rssi_mean"].max())
print("Weakest RSSI:", df["rssi_mean"].min())
print("Standard deviation:", df["rssi_mean"].std())

# --------------------------------
# GRAPH 1: RSSI DISTRIBUTION
# --------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["rssi_mean"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("RSSI (dBm)")
plt.ylabel("Number of locations")
plt.title("Distribution of Wi-Fi Signal Strength")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / "rssi_distribution.png")
plt.show()


# --------------------------------
# GRAPH 2: SPATIAL RSSI DISTRIBUTION
# --------------------------------

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    df["x_m"],
    df["y_m"],
    c=df["rssi_mean"],
    cmap="RdYlGn",
    s=100,
    edgecolors="black"
)

plt.colorbar(scatter, label="RSSI (dBm)")

plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("Spatial Distribution of Wi-Fi Signal Strength")

plt.tight_layout()
plt.savefig(BASE_DIR / "spatial_rssi_distribution.png")
plt.show()


# --------------------------------
# GRAPH 3: RSSI VARIATION
# --------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["x_m"],
    df["rssi_mean"],
    label="Average RSSI"
)

plt.xlabel("X Coordinate (m)")
plt.ylabel("RSSI (dBm)")
plt.title("Signal Strength Variation Along X Axis")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / "rssi_vs_x.png")
plt.show()

print("\nEDA COMPLETED")
