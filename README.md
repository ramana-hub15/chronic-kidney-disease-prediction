# Chronic Kidney Disease (CKD) Prediction System

An interactive web-based clinical assistant that uses machine learning (Artificial Neural Network & Logistic Regression) to analyze patient medical parameters and predict the probability of Chronic Kidney Disease (CKD).

---

## 🚀 Key Features

* **Hospital Dashboard**: At-a-glance view of total patients analyzed, CKD vs Non-CKD distributions (via interactive Pie Charts), and model performance metrics.
* **Calibrated AI Predictions**: Utilizes a calibrated Multi-Layer Perceptron (MLP) Neural Network as the main model and Logistic Regression as a secondary comparison model to output realistic, continuous risk percentages (e.g., 48.6%, 34.9%).
* **Medical Insights**: Generates diagnostic suggestions based on the severity/risk of prediction.
* **Comparison Portal**: View validation metrics (Accuracy, Precision, Recall, F1-Score) for both trained classifiers side-by-side.

---

## 🛠️ Tech Stack

* **Backend**: Flask, Flask-SQLAlchemy (SQLite database)
* **Frontend**: HTML5, CSS3 (Vanilla CSS variables & custom responsive layouts), Bootstrap 5, Font Awesome v6, Chart.js
* **Machine Learning**: Scikit-Learn (MLPClassifier, Logistic Regression, CalibratedClassifierCV, StandardScaler), Joblib, Pandas, NumPy

---

## 📋 Features Analyzed

Predictions are based on **9 clinical parameters** optimized from the clinical dataset:
1. **Serum Creatinine (sc)** - *mgs/dl*
2. **Blood Urea (bu)** - *mgs/dl*
3. **Hemoglobin (hemo)** - *gms*
4. **Packed Cell Volume (pcv)**
5. **Specific Gravity (sg)**
6. **Albumin (al)**
7. **Blood Glucose Random (bgr)** - *mgs/dl*
8. **Hypertension (htn)** - *Yes/No*
9. **Diabetes Mellitus (dm)** - *Yes/No*

---

## 💻 Setup & Installation

### 1. Clone or Open the Directory
Open your terminal/command prompt in the directory containing the project.

### 2. Install Dependencies
Make sure you have Python 3 installed. Install the required libraries:
```bash
pip install -r requirements.txt
```
*(If you do not have a requirements file, install them manually)*:
```bash
pip install flask flask-sqlalchemy pandas numpy scikit-learn joblib
```

### 3. Train the Models
Prepare the Scaler and ML models by running the training pipeline:
```bash
python train.py
```
This trains the MLP Classifier and Logistic Regression, applies sigmoid calibration, saves the `.pkl` files inside the `models/` directory, and outputs evaluation metrics.

### 4. Run the Web Server
Launch the Flask development server:
```bash
python app.py
```

### 5. Access the Portal
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔒 Default Logins
* **Username**: `admin`
* **Password**: `admin123`

---

## 📁 Directory Structure
```
ckd_/
├── app.py                  # Flask Application Server
├── train.py                # Dataset Preprocessing & Model Training Pipeline
├── kidney_disease.csv      # Clinical Dataset
├── requirements.txt        # Python Packages Setup
├── models/
│   ├── ann_model.pkl       # Calibrated MLP Neural Network
│   ├── lr_model.pkl        # Calibrated Logistic Regression
│   ├── scaler.pkl          # Fitted StandardScaler
│   └── metrics.json        # Evaluation metrics from test set
├── static/
│   └── style.css           # Custom styling guidelines
└── templates/
    ├── base.html           # Main Layout Wrapper
    ├── login.html          # Authentication view
    ├── dashboard.html      # Metrics and overview charts
    ├── input.html          # Patient health data entry form
    ├── result.html         # Diagnostic report & suggestion card
    └── compare.html        # Secondary Model evaluation graphs
```
