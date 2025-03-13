from flask import Flask, render_template, request
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scripts.feature_extraction import extract_features_from_url

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'custom_templates')

# Debugging: Print working directory and check models folder
print(f"Current Working Directory: {BASE_DIR}")
print(f"Available Templates: {os.listdir(TEMPLATES_DIR)}")
print(f"Available Model Files: {os.listdir(MODELS_DIR)}")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Load models and scaler safely
try:
    rf_model = pickle.load(open(os.path.join(MODELS_DIR, 'phishing_gb.pkl'), 'rb'))
    gb_model = pickle.load(open(os.path.join(MODELS_DIR, 'phishing_rf.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb'))
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    model_type = request.form['model']
    
    try:
        features = extract_features_from_url(url)
        features_scaled = scaler.transform([features])
    except Exception as e:
        return render_template('result.html', url=url, result=f"Error extracting features: {e}")

    if model_type == 'random_forest':
        prediction = rf_model.predict(features_scaled)[0]
    elif model_type == 'gradient_boosting':
        prediction = gb_model.predict(features_scaled)[0]
    else:
        prediction = -1

    result = 'Phishing' if prediction == 1 else 'Legitimate'
    return render_template('result.html', url=url, result=result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
