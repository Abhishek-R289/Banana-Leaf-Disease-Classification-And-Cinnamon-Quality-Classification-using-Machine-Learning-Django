# ============================================
# Cinnamon Quality Classification - FINAL CODE
# ============================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
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

print("Dataset Loaded Successfully")

# =========================
# STEP 2: Preprocessing
# =========================

# Drop ID column
if "Sample_ID" in data.columns:
    data = data.drop("Sample_ID", axis=1)

# Rename columns (clean format)
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

# Encode labels
le = LabelEncoder()
data["Quality"] = le.fit_transform(data["Quality"])

# Split features & target
X = data.drop("Quality", axis=1)
y = data["Quality"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 3: Models + Tuning
# =========================

models = {
    "Logistic Regression": (
        LogisticRegression(),
        {
            "C": [0.1, 1, 10],
            "max_iter": [500, 1000, 2000]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(),
        {
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [10, 20, None]
        }
    ),

    "SVM": (
        SVC(probability=True),
        {
            "C": [1, 10, 50],
            "gamma": ["scale", 0.01, 0.001],
            "kernel": ["rbf"]
        }
    )
}

# =========================
# STEP 4: Train Models
# =========================

accuracy_results = {}
best_model = None
best_score = 0
best_name = ""

print("\nTraining Models...\n")

for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_est = grid.best_estimator_
    preds = best_est.predict(X_test)
    acc = accuracy_score(y_test, preds)

    accuracy_results[name] = acc

    print(f"{name} Accuracy: {round(acc,4)}")
    print("Best Params:", grid.best_params_)
    print("----------------------------------")

    if acc > best_score:
        best_score = acc
        best_model = best_est
        best_name = name

# =========================
# STEP 5: Save Files
# =========================

print("\nBest Model:", best_name)
print("Best Accuracy:", best_score)

# Save best model
joblib.dump(best_model, "best_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

# Save accuracy for graph
joblib.dump(accuracy_results, "accuracy_all.pkl")

print("\nAll files saved successfully!")