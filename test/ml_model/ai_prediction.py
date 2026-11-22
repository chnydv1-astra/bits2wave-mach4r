
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load trained model and cleaned dataset
model = joblib.load(BASE_DIR / "random_forest_model.pkl")
df = pd.read_csv(BASE_DIR / "cleaned_wifi_data.csv")

# Get room boundaries from recorded coordinates
x_min, x_max = df["x_m"].min(), df["x_m"].max()
y_min, y_max = df["y_m"].min(), df["y_m"].max()

# Generate a dense prediction grid
x_values = np.linspace(x_min, x_max, 100)
y_values = np.linspace(y_min, y_max, 80)

XX, YY = np.meshgrid(x_values, y_values)

# Prepare coordinates for prediction
grid = pd.DataFrame({
    "x_m": XX.ravel(),
    "y_m": YY.ravel()
})

# Predict RSSI
predicted_rssi = model.predict(grid)

# Reshape predictions into a heatmap
RSSI_MAP = predicted_rssi.reshape(YY.shape)

# Generate visualization
plt.figure(figsize=(11, 7))

plt.imshow(
    RSSI_MAP,
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    aspect="auto",
    cmap="RdYlGn",
    vmin=-80,
    vmax=-30
)

plt.colorbar(label="Predicted RSSI (dBm)")

# Show actual measurement locations
plt.scatter(
    df["x_m"],
    df["y_m"],
    color="black",
    s=10,
    label="Measured locations"
)

plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("COVERAGE-AI - AI Predicted Wi-Fi Coverage")

plt.legend()
plt.tight_layout()

plt.savefig(BASE_DIR / "predicted_coverage_heatmap.png")
plt.show()

print("Prediction completed!")
print("Grid points:", len(grid))
print("Predicted strongest RSSI:", round(predicted_rssi.max(), 2))
print("Predicted weakest RSSI:", round(predicted_rssi.min(), 2))
