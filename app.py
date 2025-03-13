from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
from scripts.feature_extraction import extract_features_from_url
import os


app = Flask(__name__, template_folder='custom_templates')

import os

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the models directory
models_dir = os.path.join(current_dir, 'models')

# Load models
rf_model = pickle.load(open(os.path.join(models_dir, 'phishing_gb.pkl'), 'rb'))
gb_model = pickle.load(open(os.path.join(models_dir, 'phishing_rf.pkl'), 'rb'))
scaler = pickle.load(open.path.join(models_dir,'scaler.pkl'),'rb')
# rf_model = pickle.load(open('models/phishing_gb.pkl', 'rb'))
# gb_model = pickle.load(open('models/phishing_rf.pkl', 'rb'))
# scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    model_type = request.form['model']
    features = extract_features_from_url(url)
    features_scaled = scaler.transform([features])

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
