
import pandas as pd
import numpy as np
from pathlib import Path

# Locate the current script directory
BASE_DIR = Path(__file__).resolve().parent

# Load original dataset
df = pd.read_csv(BASE_DIR / "synthetic_wifi_observations.csv")

print("========== ORIGINAL DATA ==========")
print("Original observations:", len(df))

# STEP 1: Check duplicate records
print("\n========== DUPLICATE CHECK ==========")

duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

df = df.drop_duplicates()

# STEP 2: Convert numerical columns
numeric_columns = ["x_m", "y_m", "rssi_dbm"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# STEP 3: Handle missing and invalid values
print("\n========== DATA VALIDATION ==========")

print("Missing values before cleaning:")
print(df[numeric_columns].isnull().sum())

df = df.dropna(subset=numeric_columns)

# RSSI should be between -100 and 0 dBm
invalid_rssi = ~df["rssi_dbm"].between(-100, 0)

print("Invalid RSSI records:", invalid_rssi.sum())

df = df[~invalid_rssi].copy()

# STEP 4: Group repeated measurements
print("\n========== GROUPING READINGS ==========")

cleaned = df.groupby(
    ["x_m", "y_m"],
    as_index=False
).agg(
    rssi_mean=("rssi_dbm", "mean"),
    rssi_std=("rssi_dbm", "std"),
    reading_count=("rssi_dbm", "count")
)

# STEP 5: Save cleaned dataset
output_path = BASE_DIR / "cleaned_wifi_data.csv"

cleaned.to_csv(output_path, index=False)

# STEP 6: Display results
print("\n========== CLEANED DATA ==========")

print(cleaned.head(10))

print("\nOriginal observations:", len(df))
print("Unique locations:", len(cleaned))

print("\nAverage RSSI:", cleaned["rssi_mean"].mean())
print("Strongest location:", cleaned["rssi_mean"].max())
print("Weakest location:", cleaned["rssi_mean"].min())

print("\nOutput saved to:", output_path)

print("\nDATA CLEANING COMPLETED")
