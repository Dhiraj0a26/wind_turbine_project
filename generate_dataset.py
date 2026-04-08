"""
========================================
 generate_dataset.py
 Generates a synthetic dataset for Wind Turbine Gearbox Health
 25,000 rows with realistic sensor readings
========================================
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

N = 25000  # Total number of data points

print("🔄 Generating synthetic sensor dataset...")

# ---------- Sensor Features ----------

# Temperature (°C): Normal ~60-80°C; Failure risk when >90°C
temperature = np.random.normal(loc=70, scale=12, size=N)

# Vibration (mm/s): Normal ~1-5; Failure risk when >8
vibration = np.abs(np.random.normal(loc=3.5, scale=2.5, size=N))

# Pressure (bar): Normal ~4-6; Failure risk when <2 or >8
pressure = np.random.normal(loc=5.0, scale=1.5, size=N)

# Humidity (%): Normal ~30-60%; Failure risk when >75%
humidity = np.random.normal(loc=45, scale=15, size=N)

# Oil Quality (0–100 score): Higher = better; Failure risk when <30
oil_quality = np.random.normal(loc=65, scale=18, size=N)

# Clip values to realistic physical bounds
temperature  = np.clip(temperature,  20,  130)
vibration    = np.clip(vibration,     0,   20)
pressure     = np.clip(pressure,      0,   12)
humidity     = np.clip(humidity,      0,  100)
oil_quality  = np.clip(oil_quality,   0,  100)

# ---------- Failure Label Logic ----------
# A turbine is likely to FAIL if any critical condition is met

failure = (
    (temperature  > 90)  |   # Overheating
    (vibration    > 8)   |   # Excessive vibration
    (pressure     < 2)   |   # Low pressure
    (pressure     > 9)   |   # Over-pressure
    (humidity     > 75)  |   # High moisture
    (oil_quality  < 30)       # Degraded oil
).astype(int)

# ---------- Build DataFrame ----------
df = pd.DataFrame({
    "temperature":  np.round(temperature,  2),
    "vibration":    np.round(vibration,    2),
    "pressure":     np.round(pressure,     2),
    "humidity":     np.round(humidity,     2),
    "oil_quality":  np.round(oil_quality,  2),
    "failure":      failure
})

# Save to CSV
df.to_csv("sensor_data.csv", index=False)

print(f"✅ Dataset saved to sensor_data.csv")
print(f"   Total rows   : {len(df)}")
print(f"   Healthy (0)  : {(df['failure']==0).sum()}")
print(f"   Failure (1)  : {(df['failure']==1).sum()}")
print(f"   Failure Rate : {df['failure'].mean()*100:.1f}%")
print(f"\n📋 Sample rows:")
print(df.head(5).to_string(index=False))
