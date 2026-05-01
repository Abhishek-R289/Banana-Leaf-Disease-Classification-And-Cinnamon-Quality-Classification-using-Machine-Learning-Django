# 🍃 Banana Leaf Disease Classification & Cinnamon Quality Classification

### using Machine Learning and Django

---

## 📌 Overview

This project is a **web-based application** built using **Django and Machine Learning** that performs:

* 🌿 **Banana Leaf Disease Classification**
* 🌿 **Cinnamon Quality Classification**

The system uses trained machine learning models to analyze input data and provide predictions through a user-friendly web interface.

---

## 🎯 Objectives

* Detect diseases in banana leaves using dataset-based classification
* Predict the quality of cinnamon using machine learning models
* Integrate ML models into a Django web application
* Provide an easy-to-use interface for users

---

## 🚀 Technologies Used

* **Programming Language:** Python
* **Framework:** Django
* **Machine Learning:** Scikit-learn
* **Libraries:** Pandas, NumPy
* **Database:** SQLite
* **Frontend:** HTML, CSS

---

## ⚙️ Features

* 🔐 User authentication (Login / Signup / Logout)
* 🍌 Banana leaf disease prediction
* 🌿 Cinnamon quality classification
* 📊 ML model integration with Django backend
* 🌐 Web interface for user interaction

---

## 📂 Project Structure

```plaintext
Banana-Leaf-Disease-Classification/
│
├── bananaleaf/        # Django project settings
├── classifier/        # Main application logic
├── ml/                # Dataset and ML-related files
├── banana_leaf_dataset.csv
├── train_model.py
├── train_model1.py
├── manage.py
└── README.md
```

---

## 📥 Input

The system accepts:

* Dataset inputs (CSV files)
* User-provided values through web forms

---

## 📤 Output

* Predicted banana leaf disease
* Predicted cinnamon quality
* Displayed results on web interface

---

## 🧠 Machine Learning Workflow

1. Load dataset
2. Data preprocessing
3. Model training
4. Model evaluation
5. Prediction using trained model
6. Integration with Django views

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Abhishek-R289/Banana-Leaf-Disease-Classification-And-Cinnamon-Quality-Classification-using-Machine-Learning-Django.git
```

### 2️⃣ Navigate to Project

```bash
cd Banana-Leaf-Disease-Classification-And-Cinnamon-Quality-Classification-using-Machine-Learning-Django
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Apply Migrations

```bash
python manage.py migrate
```

### 5️⃣ Create Superuser (optional)

```bash
python manage.py createsuperuser
```

### 6️⃣ Run Server

```bash
python manage.py runserver
```

### 7️⃣ Open in Browser

```
http://127.0.0.1:8000/login/
```

---

## 🔐 Authentication

* Login and Signup functionality is implemented using Django authentication system
* Only authenticated users can access prediction features

---

## 📊 Future Enhancements

* Add **image-based disease detection (CNN)**
* Improve UI using Bootstrap or React
* Deploy project on cloud (AWS / Render / Railway)
* Add real-time predictions

---

## 💡 Applications

* Agriculture disease monitoring
* Quality control in spice industry
* Smart farming solutions

---

## 👨‍💻 Author

**Abhishek R**

---

## 📎 GitHub Repository

Project available at:
https://github.com/Abhishek-R289/Banana-Leaf-Disease-Classification-And-Cinnamon-Quality-Classification-using-Machine-Learning-Django

---
