from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

# Load model and datasets
model = joblib.load('model.joblib')

# Load datasets
try:
    df_dataset = pd.read_csv('dataset.csv')
    df_symptoms = pd.read_csv('symptom_severity.csv')
    df_description = pd.read_csv('symptom_description.csv')
    df_precaution = pd.read_csv('symptom_precaution.csv')
except Exception as e:
    print(f"Error loading CSV files: {e}")
    # Create placeholder dataframes if files don't exist
    df_dataset = pd.DataFrame()
    df_symptoms = pd.DataFrame()
    df_description = pd.DataFrame()
    df_precaution = pd.DataFrame()

# Prepare symptom list
if not df_symptoms.empty:
    symptom_list = df_symptoms['Symptom'].tolist()
else:
    # Create default symptom list from dataset if available
    if not df_dataset.empty:
        symptom_list = []
        for col in df_dataset.columns:
            if col.startswith('Symptom_'):
                symptoms = df_dataset[col].dropna().unique()
                symptom_list.extend(symptoms)
        symptom_list = list(set(symptom_list))
    else:
        symptom_list = []

# Clean symptom names for consistency
def clean_symptom_name(symptom):
    """Clean symptom name for matching"""
    if isinstance(symptom, str):
        return symptom.lower().strip().replace('_', ' ').replace('-', ' ')
    return ""

# Clean all symptom lists
symptom_list = [clean_symptom_name(s) for s in symptom_list if s]
symptom_list = sorted(list(set(symptom_list)))

# Create symptom index mapping
symptom_index = {symptom: idx for idx, symptom in enumerate(symptom_list)}

# Function to prepare input for prediction
def prepare_input(selected_symptoms):
    """Convert selected symptoms to model input format"""
    # Check if we have symptom severity data
    if df_symptoms.empty or 'Symptom' not in df_symptoms.columns or 'weight' not in df_symptoms.columns:
        # Fallback: create a simple binary encoding
        input_vector = [0] * len(symptom_list)
        for symptom in selected_symptoms:
            clean_symp = clean_symptom_name(symptom)
            if clean_symp in symptom_index:
                input_vector[symptom_index[clean_symp]] = 1
        return np.array([input_vector], dtype=np.float64)
    
    # Original logic with symptom weights
    a = np.array(df_symptoms["Symptom"])
    b = np.array(df_symptoms["weight"])
    
    # Clean the selected symptoms for matching
    selected_symptoms_clean = [clean_symptom_name(s) for s in selected_symptoms]
    
    # Replace symptoms with their weights
    weighted_symptoms = []
    for symptom in selected_symptoms_clean:
        found = False
        for k in range(len(a)):
            if clean_symptom_name(a[k]) == symptom:
                weighted_symptoms.append(float(b[k]))  # Convert to float
                found = True
                break
        if not found:
            weighted_symptoms.append(1.0)  # Default weight if not found
    
    MAX_LEN = 17
    
    if len(weighted_symptoms) < MAX_LEN:
        weighted_symptoms.extend([0.0] * (MAX_LEN - len(weighted_symptoms)))
    elif len(weighted_symptoms) > MAX_LEN:
        weighted_symptoms = weighted_symptoms[:MAX_LEN]
    
    return np.array([weighted_symptoms], dtype=np.float64)

# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to Python scalar
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Function to get disease description
def get_disease_description(disease_name):
    """Get description of the disease"""
    if df_description.empty:
        return f"Information about {disease_name}"
    
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_description.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            return str(row['Description'])
    
    return f"No detailed description available for {disease_name}"

# Function to get disease precautions
def get_disease_precautions(disease_name):
    """Get precautions for the disease"""
    if df_precaution.empty:
        return ["Consult a doctor", "Take prescribed medication", "Get adequate rest", "Maintain good hygiene"]
    
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_precaution.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            precautions = []
            for i in range(1, 5):  # Assuming 4 precaution columns
                col_name = f'Precaution_{i}'
                if col_name in row and pd.notna(row[col_name]):
                    precautions.append(str(row[col_name]))
            return precautions
    
    return ["Consult a doctor", "Take prescribed medication", "Get adequate rest", "Maintain good hygiene"]

# Function to get disease severity
def get_disease_severity(disease_name):
    """Estimate disease severity based on symptoms"""
    if df_symptoms.empty or df_dataset.empty:
        return "Medium"
    
    # Count number of symptoms for the disease
    disease_symptoms_count = 0
    disease_name_clean = str(disease_name).strip().lower()
    
    for idx, row in df_dataset.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            disease_symptoms_count = sum(1 for col in df_dataset.columns if col.startswith('Symptom_') and pd.notna(row[col]))
            break
    
    if disease_symptoms_count > 10:
        return "High"
    elif disease_symptoms_count > 5:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle disease prediction"""
    try:
        # Get symptoms from form
        selected_symptoms = request.form.getlist('symptoms[]')
        
        if not selected_symptoms:
            return jsonify({
                'error': 'Please select at least one symptom'
            })
        
        print(f"Selected symptoms: {selected_symptoms}")
        
        # Prepare input for model
        input_data = prepare_input(selected_symptoms)
        print(f"Input data shape: {input_data.shape}")
        print(f"Input data type: {type(input_data)}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        print(f"Prediction: {prediction}")
        print(f"Probabilities shape: {probabilities.shape}")
        
        # Get top 3 predictions
        top_n = 3
        classes = model.classes_
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_diseases = classes[top_indices]
        top_probabilities = probabilities[top_indices]
        
        print(f"Top diseases: {top_diseases}")
        print(f"Top probabilities: {top_probabilities}")
        
        # Prepare response
        predictions = []
        for disease, prob in zip(top_diseases, top_probabilities):
            predictions.append({
                'disease': str(disease),  # Ensure it's a string
                'probability': float(round(prob * 100, 2)),  # Convert to float
                'description': get_disease_description(disease),
                'precautions': get_disease_precautions(disease),
                'severity': get_disease_severity(disease)
            })
        
        # Get recommended doctor type (simplified)
        doctor_types = {
            'dermatologist': [
                'Acne', 'Psoriasis', 'Fungal infection', 'Impetigo', 'Chicken pox'
            ],
            'endocrinologist': [
                'Hyperthyroidism', 'Hypothyroidism', 'Diabetes', 'Hypoglycemia'
            ],
            'infectious disease specialist': [
                'AIDS', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
                'Tuberculosis', 'Typhoid', 'Dengue', 'Malaria', 'Pneumonia'
            ],
            'gastroenterologist': [
                'Chronic cholestasis', 'Jaundice', 'Peptic ulcer diseae', 'GERD', 'Alcoholic hepatitis', 'Gastroenteritis'
            ],
            'cardiologist': [
                'Hypertension', 'Heart attack'
            ],
            'neurologist': [
                'Migraine', '(vertigo) Paroymsal  Positional Vertigo', 'Paralysis (brain hemorrhage)'
            ],
            'rheumatologist': [
                'Arthritis', 'Osteoarthristis', 'Cervical spondylosis', 'Varicose veins', 'Bronchial Asthma'
            ],
            'urologist': [
                'Urinary tract infection'
            ],
            'allergist': [
                'Allergy', 'Common Cold'
            ],
            'general physician': [
                'Drug Reaction'  # Default for any disease not clearly under a specialty
            ]
        }
        
        primary_disease = str(predictions[0]['disease']).lower()
        recommended_doctor = 'general physician'
        
        for doctor, keywords in doctor_types.items():
            if any(keyword in primary_disease for keyword in keywords):
                recommended_doctor = doctor
                break
        
        # Prepare response data with converted types
        response_data = {
            'success': True,
            'predictions': convert_numpy_types(predictions),
            'selected_symptoms': selected_symptoms,
            'recommended_doctor': recommended_doctor.capitalize()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict(): {e}")
        print(f"Error details:\n{error_details}")
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'details': str(error_details) if app.debug else None
        })

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get all available symptoms (for API)"""
    return jsonify({
        'symptoms': symptom_list,
        'count': len(symptom_list)
    })

@app.route('/search_symptoms', methods=['POST'])
def search_symptoms():
    """Search for symptoms based on query"""
    try:
        data = request.get_json()
        query = data.get('query', '').lower() if data else ''
        
        if not query:
            return jsonify({'symptoms': []})
        
        matched_symptoms = [s for s in symptom_list if query in s.lower()]
        
        return jsonify({
            'symptoms': matched_symptoms[:10]  # Limit to 10 results
        })
    except Exception as e:
        return jsonify({'symptoms': [], 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)