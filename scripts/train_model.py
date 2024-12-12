import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

# Load the dataset
data = pd.read_csv('Phishing_Legitimate_full.csv')

# Select the required columns
selected_columns = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
    'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
    'NumPercent', 'NumQueryComponents', 'NumNumericChars', 'HostnameLength',
    'PathLength', 'QueryLength'
]
X = data[selected_columns]
y = data['CLASS_LABEL']

# Handle missing values
X.fillna(X.median(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Save the trained models and scaler
with open('models/phishing_rf.pkl', 'wb') as rf_file:
    pickle.dump(rf_model, rf_file)

with open('models/phishing_gb.pkl', 'wb') as gb_file:
    pickle.dump(gb_model, gb_file)

with open('models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Models and scaler have been saved successfully!")
