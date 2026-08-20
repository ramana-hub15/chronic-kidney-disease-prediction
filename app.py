from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import numpy as np
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Base directory of this script (so model paths work regardless of CWD)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'supersecretkey_ckd'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ckd_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load Models and Scaler
ann_model = None
lr_model = None
scaler = None
metrics = {}

try:
    ann_model = joblib.load(os.path.join(BASE_DIR, 'models', 'ann_model.pkl'))
    print("ANN model loaded successfully")

    lr_model = joblib.load(os.path.join(BASE_DIR, 'models', 'lr_model.pkl'))
    print("LR model loaded successfully")

    scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
    print("Scaler loaded successfully:", type(scaler))

    with open(os.path.join(BASE_DIR, 'models', 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    print("Metrics loaded successfully")

except Exception as e:
    print(f"ERROR loading models: {e}")
    import traceback
    traceback.print_exc()
# --- Database Models ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='Doctor')

class PatientPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sc = db.Column(db.Float, nullable=False)
    bu = db.Column(db.Float, nullable=False)
    hemo = db.Column(db.Float, nullable=False)
    pcv = db.Column(db.Float, nullable=False)
    sg = db.Column(db.Float, nullable=False)
    al = db.Column(db.Float, nullable=False)
    bgr = db.Column(db.Float, nullable=False)
    htn = db.Column(db.Integer, nullable=False)
    dm = db.Column(db.Integer, nullable=False)
    prediction_ann = db.Column(db.Integer, nullable=False)
    probability_ann = db.Column(db.Float, nullable=False)
    prediction_lr = db.Column(db.Integer, nullable=False)
    probability_lr = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()
    # Create default admin if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', password=generate_password_hash('admin123'), role='Admin')
        db.session.add(admin)
        db.session.commit()

# --- Helpers ---

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_predictions = PatientPrediction.query.count()
    ckd_cases = PatientPrediction.query.filter_by(prediction_ann=1).count()
    non_ckd_cases = total_predictions - ckd_cases
    return render_template('dashboard.html', 
                           total=total_predictions, 
                           ckd=ckd_cases, 
                           non_ckd=non_ckd_cases, 
                           metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Guard: check if models are loaded
            if scaler is None or ann_model is None or lr_model is None:
                flash('Models are not loaded. Please run train.py first to train and save the models.', 'danger')
                return redirect(url_for('predict'))

            # Extract features in exact order used during training
            # ['sc', 'bu', 'hemo', 'pcv', 'sg', 'al', 'bgr', 'htn', 'dm']
            sc = float(request.form['sc'])
            bu = float(request.form['bu'])
            hemo = float(request.form['hemo'])
            pcv = float(request.form['pcv'])
            sg = float(request.form['sg'])
            al = float(request.form['al'])
            bgr = float(request.form['bgr'])
            htn = int(request.form['htn'])
            dm = int(request.form['dm'])
            
            features = pd.DataFrame([[sc, bu, hemo, pcv, sg, al, bgr, htn, dm]], 
                                    columns=['sc', 'bu', 'hemo', 'pcv', 'sg', 'al', 'bgr', 'htn', 'dm'])
            
            features_scaled = scaler.transform(features)
            
            # ANN Prediction
            ann_pred = int(ann_model.predict(features_scaled)[0])
            ann_prob_raw = float(ann_model.predict_proba(features_scaled)[0][1])
            
            # Apply temperature scaling to smooth/spread out the probability values
            # Using T = 2.0 to soften predictions and make them more continuous/granular
            # logits = log(p / (1 - p))
            if ann_prob_raw <= 0:
                ann_prob_raw = 0.0001
            elif ann_prob_raw >= 1:
                ann_prob_raw = 0.9999
            
            logits_ann = np.log(ann_prob_raw / (1 - ann_prob_raw))
            ann_prob = float(1 / (1 + np.exp(-logits_ann / 2.0)))
            
            # LR Prediction
            lr_pred = int(lr_model.predict(features_scaled)[0])
            lr_prob_raw = float(lr_model.predict_proba(features_scaled)[0][1])
            
            if lr_prob_raw <= 0:
                lr_prob_raw = 0.0001
            elif lr_prob_raw >= 1:
                lr_prob_raw = 0.9999
                
            logits_lr = np.log(lr_prob_raw / (1 - lr_prob_raw))
            lr_prob = float(1 / (1 + np.exp(-logits_lr / 2.0)))
            
            # Save to DB
            record = PatientPrediction(
                sc=sc, bu=bu, hemo=hemo, pcv=pcv, sg=sg, al=al, bgr=bgr, htn=htn, dm=dm,
                prediction_ann=ann_pred, probability_ann=ann_prob,
                prediction_lr=lr_pred, probability_lr=lr_prob
            )
            db.session.add(record)
            db.session.commit()
            
            return redirect(url_for('result', id=record.id))
            
        except Exception as e:
            flash(f'Error making prediction: {e}', 'danger')
            return redirect(url_for('predict'))
            
    return render_template('input.html')

@app.route('/result/<int:id>')
@login_required
def result(id):
    record = PatientPrediction.query.get_or_404(id)
    
    # Determine risk level
    prob = record.probability_ann
    if prob < 0.3:
        risk = "Low Risk"
        color = "success"
    elif prob < 0.7:
        risk = "Moderate Risk"
        color = "warning"
    else:
        risk = "High Risk"
        color = "danger"
        
    return render_template('result.html', record=record, risk=risk, color=color)

@app.route('/compare')
@login_required
def compare():
    return render_template('compare.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
