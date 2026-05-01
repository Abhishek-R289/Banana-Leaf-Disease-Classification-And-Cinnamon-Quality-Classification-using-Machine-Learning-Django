import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -----------------------------
# Create static folder
# -----------------------------
os.makedirs("classifier/static", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("banana_leaf_dataset.csv").dropna()

# -----------------------------
# Encode categorical
# -----------------------------
le = LabelEncoder()
df['ColorIntensity'] = le.fit_transform(df['ColorIntensity'])
df['Texture'] = le.fit_transform(df['Texture'])
df['SoilType'] = le.fit_transform(df['SoilType'])
df['DiseaseLabel'] = le.fit_transform(df['DiseaseLabel'])

# -----------------------------
# Split data
# -----------------------------
X = df.drop("DiseaseLabel", axis=1)
y = df["DiseaseLabel"]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "classifier/scaler.pkl")

# -----------------------------
# Train test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "rf": RandomForestClassifier(n_estimators=300, max_depth=12),

    "lr": LogisticRegression(max_iter=2000),   # 🔥 increase

    "knn": KNeighborsClassifier(n_neighbors=3),

    "svm": SVC(kernel='rbf', probability=True, max_iter=2000)  # 🔥 add this
}

accuracy = {}

# -----------------------------
# Train models
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracy[name] = acc
    print(f"{name} accuracy: {acc}")

# -----------------------------
# Save models
# -----------------------------
for name, model in models.items():
    joblib.dump(model, f"classifier/{name}.pkl")

joblib.dump(accuracy, "classifier/accuracy.pkl")

# -----------------------------
# 📊 Accuracy Graph
# -----------------------------
plt.figure()
plt.bar(accuracy.keys(), accuracy.values())
plt.title("Model Accuracy")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.savefig("classifier/static/accuracy.png")
plt.close()

# -----------------------------
# 📉 Confusion Matrix
# -----------------------------
best_model_name = max(accuracy, key=accuracy.get)
best_model = models[best_model_name]

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("classifier/static/confusion_matrix.png")
plt.close()

print("✅ Training + Graphs Complete!")