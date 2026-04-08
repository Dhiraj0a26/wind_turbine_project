"""
========================================
 train_model.py
 Trains a Random Forest Classifier on the
 wind turbine sensor dataset and saves the model.
========================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("="*55)
print("  Wind Turbine Predictive Maintenance — Model Training")
print("="*55)

# ── Step 1: Load Data ──────────────────────────────────────
print("\n📂 Step 1: Loading dataset...")

# Auto-generate if CSV doesn't exist yet
if not os.path.exists("sensor_data.csv"):
    print("   sensor_data.csv not found → generating now...")
    exec(open("generate_dataset.py").read())

df = pd.read_csv("sensor_data.csv")
print(f"   Loaded {len(df)} rows, {df.shape[1]} columns")

# ── Step 2: Feature / Label Split ─────────────────────────
print("\n🔧 Step 2: Splitting features and labels...")

FEATURES = ["temperature", "vibration", "pressure", "humidity", "oil_quality"]
TARGET   = "failure"

X = df[FEATURES]
y = df[TARGET]

print(f"   Features : {FEATURES}")
print(f"   Class balance → Healthy: {(y==0).sum()}  |  Failure: {(y==1).sum()}")

# ── Step 3: Train-Test Split ───────────────────────────────
print("\n✂️  Step 3: Train-test split (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# ── Step 4: Feature Scaling ────────────────────────────────
print("\n📏 Step 4: Scaling features with StandardScaler...")

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Step 5: Train Model ────────────────────────────────────
print("\n🌲 Step 5: Training Random Forest Classifier...")
print("   (100 trees, this may take a few seconds...)")

model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=12,       # Limit tree depth to prevent overfitting
    random_state=42,
    n_jobs=-1           # Use all CPU cores for speed
)
model.fit(X_train_scaled, y_train)
print("   ✅ Training complete!")

# ── Step 6: Evaluate ──────────────────────────────────────
print("\n📊 Step 6: Evaluating on test set...")

y_pred    = model.predict(X_test_scaled)
accuracy  = accuracy_score(y_test, y_pred)

print(f"\n{'─'*45}")
print(f"  🎯 Test Accuracy : {accuracy*100:.2f}%")
print(f"{'─'*45}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Healthy (0)", "Failure (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("🔲 Confusion Matrix:")
print(f"   True Healthy predicted Healthy : {cm[0][0]}")
print(f"   True Healthy predicted Failure : {cm[0][1]}  ← False Alarm")
print(f"   True Failure predicted Healthy : {cm[1][0]}  ← Missed Failure")
print(f"   True Failure predicted Failure : {cm[1][1]}")

# ── Step 7: Feature Importance ────────────────────────────
print("\n🏆 Feature Importance (higher = more influential):")
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"   {feat:<15} {imp:.4f}  {bar}")

# ── Step 8: Save Model & Scaler ───────────────────────────
print("\n💾 Step 7: Saving model and scaler...")

joblib.dump(model,  "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("   Saved → rf_model.pkl")
print("   Saved → scaler.pkl")
print("\n✅ All done! The model is ready for the API.")
print("="*55)
