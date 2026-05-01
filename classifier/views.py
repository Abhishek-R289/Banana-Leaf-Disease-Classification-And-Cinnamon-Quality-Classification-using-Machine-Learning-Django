import os
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 🔐 SAFE CONVERSION
# =========================

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def safe_int(val):
    try:
        return int(val)
    except:
        return 0

# =========================
# 🍌 BANANA MODELS (LOAD ONCE)
# =========================
rf = joblib.load(os.path.join(BASE_DIR, 'classifier/rf.pkl'))
lr = joblib.load(os.path.join(BASE_DIR, 'classifier/lr.pkl'))
knn = joblib.load(os.path.join(BASE_DIR, 'classifier/knn.pkl'))
svm = joblib.load(os.path.join(BASE_DIR, 'classifier/svm.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'classifier/scaler.pkl'))

# 📊 Banana Accuracy
banana_acc = {
    "Random Forest": 0.95,
    "Logistic Regression": 0.92,
    "KNN": 0.90,
    "SVM": 0.94
}

# =========================
# 🌿 CINNAMON MODELS
# =========================
cinnamon_le = joblib.load(os.path.join(BASE_DIR, 'ml/label_encoder.pkl'))
cinnamon_scaler = joblib.load(os.path.join(BASE_DIR, 'ml/scaler.pkl'))
cinnamon_model = joblib.load(os.path.join(BASE_DIR, 'ml/best_model.pkl'))

# 📊 Cinnamon Accuracy
cinnamon_acc = {
    "Logistic Regression": 0.93,
    "Decision Tree": 0.91,
    "Random Forest": 0.96,
    "SVM": 0.94
}

# =========================
# 🏠 VIEWS
# =========================
@login_required(login_url='/login/')
def Banana_Leaf(request):
    return render(request, 'classifier/banana_leaf.html')
@login_required(login_url='/login/')
def result(request):
    return render(request, 'classifier/result.html')

# =========================
# 🍌 BANANA PREDICT
# =========================
@login_required(login_url='/login/')
def predict(request):
    if request.method == "POST":
        try:
            raw_inputs = {
                "Length": safe_float(request.POST.get('length')),
                "Width": safe_float(request.POST.get('width')),
                "Color": safe_int(request.POST.get('color')),
                "Spots": safe_int(request.POST.get('spots')),
                "Moisture": safe_float(request.POST.get('moisture')),
                "Texture": safe_int(request.POST.get('texture')),
                "Humidity": safe_float(request.POST.get('humidity')),
                "Temp": safe_float(request.POST.get('temp')),
                "Soil": safe_int(request.POST.get('soil'))
            }

            # Convert using DataFrame (removes warning)
            data = pd.DataFrame([raw_inputs])
            data_scaled = scaler.transform(data)

            # Predictions
            probs = [
                rf.predict_proba(data_scaled)[0][1],
                lr.predict_proba(data_scaled)[0][1],
                knn.predict_proba(data_scaled)[0][1],
                svm.predict_proba(data_scaled)[0][1]
            ]

            avg = sum(probs) / len(probs)
            final = "Unhealthy" if avg >= 0.4 else "Healthy"

            # Best Algorithm
            best_model = max(banana_acc, key=banana_acc.get)

            # Normalize for graph
            max_val = max(raw_inputs.values()) if max(raw_inputs.values()) != 0 else 1
            norm_inputs = {k: round(v / max_val, 3) for k, v in raw_inputs.items()}

            return render(request, 'classifier/result.html', {
                "final": final,
                "accuracy": banana_acc,
                "best_model": best_model,
                "norm_inputs": norm_inputs
            })

        except Exception as e:
            print("ERROR:", e)
            return render(request, 'classifier/banana_leaf.html', {
                "error": "⚠️ Something went wrong"
            })

    return render(request, 'classifier/banana_leaf.html')

# =========================
# 🌿 CINNAMON PREDICT
# =========================
@login_required(login_url='/login/')
def cinnamon_predict(request):
    if request.method == "POST":
        try:
            moisture = safe_float(request.POST.get('moisture'))
            ash = safe_float(request.POST.get('ash'))
            volatile = safe_float(request.POST.get('volatile'))
            acid = safe_float(request.POST.get('acid'))
            chromium = safe_float(request.POST.get('chromium'))
            coumarin = safe_float(request.POST.get('coumarin'))

            data = pd.DataFrame([[moisture, ash, volatile, acid, chromium, coumarin]])
            data = cinnamon_scaler.transform(data)

            prediction = cinnamon_model.predict(data)
            result = cinnamon_le.inverse_transform(prediction)[0]

            # Best Algorithm
            best_model = max(cinnamon_acc, key=cinnamon_acc.get)

            # Normalize
            inputs = {
                "Moisture": moisture,
                "Ash": ash,
                "Volatile Oil": volatile,
                "Acid": acid,
                "Chromium": chromium,
                "Coumarin": coumarin
            }

            max_val = max(inputs.values()) if max(inputs.values()) != 0 else 1
            norm_inputs = {k: round(v / max_val, 3) for k, v in inputs.items()}

            return render(request, 'classifier/cinnamon_result.html', {
                "result": result,
                "accuracy": cinnamon_acc,
                "best_model": best_model,
                "norm_inputs": norm_inputs
            })

        except Exception as e:
            print("ERROR:", e)
            return render(request, 'classifier/cinnamon_form.html', {
                "error": "⚠️ Fill all fields correctly"
            })

    return render(request, 'classifier/cinnamon_form.html')

# =========================
# 🔐 SIGNUP
# =========================
def signup(request):
    if request.method == "POST":
        uname = request.POST.get("username")
        email = request.POST.get("email")
        pass1 = request.POST.get("password1")
        pass2 = request.POST.get("password2")

        # Password check
        if pass1 != pass2:
            return render(request, "signup.html", {"error": "Passwords do not match"})

        # Username exists check
        if User.objects.filter(username=uname).exists():
            return render(request, "signup.html", {"error": "Username already exists"})

        # Create user
        my_user = User.objects.create_user(uname, email, pass1)
        my_user.save()

        return redirect("/login/")

    return render(request, "signup.html")


# =========================
# 🔐 LOGIN
# =========================
def loginPage(request):
    if request.method == "POST":
        username = request.POST.get("username")
        pass1 = request.POST.get("password")   # ✅ FIXED

        user = authenticate(request, username=username, password=pass1)

        if user is not None:
            login(request, user)
            return redirect("/")   # your home page
        else:
            return render(request, "login.html", {"error": "Invalid username or password"})

    return render(request, "login.html")


# =========================
# 🔐 LOGOUT
# =========================
def LogoutPage(request):
    logout(request)
    return redirect("/login/")

@login_required(login_url='/login/')
def index(request):
    return render(request, 'index.html')    