import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('common_diseases_dataset.csv')

# Encode the symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(data['Symptoms'].apply(lambda x: x.split(',')))

# Encode target variables
y_disease = data['Disease Name']
y_severity = data['Severity']
y_treatment = data['Immediate Fast Response Treatment']

# Split the data
X_train, X_test, y_train_disease, y_test_disease = train_test_split(X, y_disease, test_size=0.2, random_state=42)
_, _, y_train_severity, y_test_severity = train_test_split(X, y_severity, test_size=0.2, random_state=42)
_, _, y_train_treatment, y_test_treatment = train_test_split(X, y_treatment, test_size=0.2, random_state=42)

# Train the decision tree models
model_disease = DecisionTreeClassifier().fit(X_train, y_train_disease)
model_severity = DecisionTreeClassifier().fit(X_train, y_train_severity)
model_treatment = DecisionTreeClassifier().fit(X_train, y_train_treatment)

# Save models
joblib.dump(model_disease, 'model_disease.pkl')
joblib.dump(model_severity, 'model_severity.pkl')
joblib.dump(model_treatment, 'model_treatment.pkl')
joblib.dump(mlb, 'mlb.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "NHealth API - Predict diseases from symptoms."

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    try:
        symptoms = request.json['symptoms']
        symptoms_encoded = mlb.transform([symptoms.split(',')])
        
        disease_pred = model_disease.predict(symptoms_encoded)
        severity_pred = model_severity.predict(symptoms_encoded)
        treatment_pred = model_treatment.predict(symptoms_encoded)
        
        response = {
            'disease': disease_pred[0],
            'severity': severity_pred[0],
            'treatment': treatment_pred[0]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
