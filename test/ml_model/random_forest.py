
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. SET PROJECT DIRECTORY
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

print("Project directory:", BASE_DIR)

# --------------------------------------------------
# 2. LOAD CLEANED DATA
# --------------------------------------------------

data_path = BASE_DIR / "cleaned_wifi_data.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

df = pd.read_csv(data_path)

print("\nDataset loaded successfully!")
print("Total samples:", len(df))

# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

X = df[["x_m", "y_m"]]
y = df["rssi_mean"]

# --------------------------------------------------
# 4. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. CREATE AND TRAIN RANDOM FOREST
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 6. SAVE MODEL IMMEDIATELY
# --------------------------------------------------

model_path = BASE_DIR / "random_forest_model.pkl"

joblib.dump(model, model_path)

if model_path.exists():
    print("\nMODEL SAVED SUCCESSFULLY!")
    print("Saved location:", model_path)
    print("File size:", model_path.stat().st_size, "bytes")
else:
    raise RuntimeError("Model file was not created!")

# --------------------------------------------------
# 7. PREDICT TEST DATA
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE:", round(mae, 2), "dB")
print("RMSE:", round(rmse, 2), "dB")
print("R2 Score:", round(r2, 3))
print("=======================================")

# --------------------------------------------------
# 9. ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.8
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect prediction"
)

plt.xlabel("Actual RSSI (dBm)")
plt.ylabel("Predicted RSSI (dBm)")
plt.title("Random Forest: Actual vs Predicted")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

graph_path = BASE_DIR / "model_evaluation.png"

plt.savefig(graph_path, dpi=300)

print("\nEvaluation graph saved:", graph_path)

plt.show()

print("\n========== PROCESS COMPLETE ==========")
