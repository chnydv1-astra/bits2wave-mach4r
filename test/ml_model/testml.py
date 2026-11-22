
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("test/ml_model/synthetic_wifi_observations.csv")

# 1. Display first five records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# 2. Dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Column information
print("\n========== COLUMN INFORMATION ==========")
print(df.info())

# 4. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 5. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Unique measurement locations
print("\n========== UNIQUE LOCATIONS ==========")
print(df[["x_m", "y_m"]].drop_duplicates().shape[0])

# 7. RSSI range
print("\n========== RSSI RANGE ==========")
print("Minimum RSSI:", df["rssi_dbm"].min())
print("Maximum RSSI:", df["rssi_dbm"].max())
print("Average RSSI:", df["rssi_dbm"].mean())

print("\nDATASET INSPECTION COMPLETED")
