# train_model.py

import pandas as pd
import joblib

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =========================
# STEP 1: Load Dataset
# =========================
data = pd.read_csv("balanced_cinnamon_quality_dataset.csv")

print("Dataset Loaded Successfully\n")
print(data.head())

# =========================
# STEP 2: Preprocessing
# =========================

# Drop unnecessary column
data = data.drop("Sample_ID", axis=1)

# Rename columns (IMPORTANT FIX)
data.columns = [
    "Moisture",
    "Ash",
    "Volatile_Oil",
    "Acid_Insoluble_Ash",
    "Chromium",
    "Coumarin",
    "Quality"
]

# Remove missing values
data = data.dropna()

# Encode target
le = LabelEncoder()
data["Quality"] = le.fit_transform(data["Quality"])

# Split features & target
X = data.drop("Quality", axis=1)
y = data["Quality"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# STEP 3: Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 4: Models (4 Algorithms)
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel='rbf', probability=True)
}

# =========================
# STEP 5: Train & Evaluate
# =========================
accuracy_results = {}
trained_models = {}

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    accuracy_results[name] = acc
    trained_models[name] = model

    print(f"{name} Accuracy: {acc:.4f}")

# =========================
# STEP 6: Best Model
# =========================
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", accuracy_results[best_model_name])

# =========================
# STEP 7: Save Files
# =========================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nSaved Successfully:")
print("✔ best_model.pkl")
print("✔ scaler.pkl")
print("✔ label_encoder.pkl")