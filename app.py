from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model and encoders
try:
    loaded_model = joblib.load('model.joblib')
    loaded_symptom_index = joblib.load('symptom_index.joblib')
    loaded_label_encoder = joblib.load('label_encoder.joblib')
    print("Trained model and encoders loaded successfully")
except Exception as e:
    print(f"Error loading trained files: {e}")
    loaded_model = None
    loaded_symptom_index = {}
    loaded_label_encoder = None

# Load datasets
try:
    df_dataset = pd.read_csv('dataset.csv') 
    df_description = pd.read_csv('disease_description.csv')
    df_precaution = pd.read_csv('disease_precaution.csv')
    df_disease_severity = pd.read_csv('disease_severity.csv')
    df_symptom_severity = pd.read_csv('symptom_severity.csv')
except Exception as e:
    print(f"Error loading CSV files: {e}")
    df_dataset = pd.DataFrame()
    df_description = pd.DataFrame()
    df_precaution = pd.DataFrame()
    df_disease_severity = pd.DataFrame()
    df_symptom_severity = pd.DataFrame()

# Create severity map from severity dataset
if not df_symptom_severity.empty:
    severity_df = df_symptom_severity.copy()
    severity_df['Symptom'] = (
        severity_df['Symptom']
        .str.lower()
        .str.strip()
        .str.replace('_', ' ')
    )
    severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))
else:
    severity_map = {}

# Prepare symptom list from trained index
if loaded_symptom_index:
    symptom_list = list(loaded_symptom_index.keys())
    symptom_list = sorted(symptom_list)
    print(f"Loaded {len(symptom_list)} symptoms from trained index")
else:
    # Fallback: create symptom list from datasets
    symptom_list = []
    if not df_symptom_severity.empty:
        symptom_list = df_symptom_severity['Symptom'].tolist()
    elif not df_dataset.empty:
        for col in df_dataset.columns:
            if col.startswith('Symptom_'):
                symptoms = df_dataset[col].dropna().unique()
                symptom_list.extend(symptoms)
        symptom_list = list(set(symptom_list))
    
    # Clean symptom names
    def clean_symptom_name(symptom):
        if isinstance(symptom, str):
            return symptom.lower().strip().replace('_', ' ')
        return ""
    
    symptom_list = [clean_symptom_name(s) for s in symptom_list if s]
    symptom_list = sorted(list(set(symptom_list)))
    print(f"Loaded {len(symptom_list)} symptoms from datasets")

# Function to prepare input for prediction
def prepare_input(selected_symptoms):
    """Convert selected symptoms to model input format using trained encoders"""
    if not loaded_symptom_index:
        raise ValueError("Trained symptom index not loaded")
    
    # Create zero vector with same length as training symptoms
    x = np.zeros(len(loaded_symptom_index))
    
    # Process each symptom
    for symptom in selected_symptoms:
        # Clean the symptom name (same as training)
        s = symptom.lower().strip().replace('_', ' ')
        
        # Check if symptom exists in trained index
        if s in loaded_symptom_index:
            # Use severity weight if available, otherwise 1
            weight = severity_map.get(s, 1)
            x[loaded_symptom_index[s]] = weight
    
    return x.reshape(1, -1)  # Reshape to (1, n_features)

# Keep all your helper functions unchanged:
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def get_disease_description(disease_name):
    """Get description of the disease"""
    if df_description.empty:
        return f"Information about {disease_name}"
    
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_description.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            return str(row['Description'])
    
    return f"No detailed description available for {disease_name}"

def get_disease_precautions(disease_name):
    """Get precautions for the disease"""
    if df_precaution.empty:
        return ["Consult a doctor", "Take prescribed medication", "Get adequate rest", "Maintain good hygiene"]
    
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_precaution.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            precautions = []
            for i in range(1, 5):
                col_name = f'Precaution_{i}'
                if col_name in row and pd.notna(row[col_name]):
                    precautions.append(str(row[col_name]))
            return precautions
    
    return ["Consult a doctor", "Take prescribed medication", "Get adequate rest", "Maintain good hygiene"]

def get_disease_severity(disease_name):
    """Show disease severity"""
    disease_name = disease_name.lower().strip()
    
    match = df_disease_severity[df_disease_severity["Disease"] == disease_name]
    
    if not match.empty:
        return match.iloc[0]["Severity"]
    
    return "Medium"  # default fallback

# Routes (keep your existing routes with updated predict function)
@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle disease prediction using trained model"""
    try:
        if not loaded_model:
            return jsonify({
                'error': 'Prediction model not loaded. Please check server configuration.'
            })
        
        # Get symptoms from form
        selected_symptoms = request.form.getlist('symptoms[]')
        
        if not selected_symptoms:
            return jsonify({
                'error': 'Please select at least one symptom'
            })
        
        print(f"Selected symptoms: {selected_symptoms}")
        
        # Prepare input using trained encoders
        input_data = prepare_input(selected_symptoms)
        
        # Make prediction using trained model
        probabilities = loaded_model.predict_proba(input_data)[0]
        
        # Get top 3 predictions with confidence
        top_n = 3
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_probs = probabilities[top_indices]
        
        # Normalize probabilities to sum to 100%
        normalized_probs = top_probs / top_probs.sum()
        
        # Prepare predictions with confidence scores
        predictions = []
        for idx, prob in zip(top_indices, normalized_probs):
            disease_name = loaded_label_encoder.inverse_transform([idx])[0]
            confidence = round(prob * 100, 2)
            
            predictions.append({
                'disease': str(disease_name).title(),
                'confidence': float(confidence),
                'description': get_disease_description(disease_name),
                'precautions': get_disease_precautions(disease_name),
                'severity': get_disease_severity(disease_name)
            })
        
        # Get recommended doctor type
        doctor_types = {
            'dermatologist': ['Acne', 'Psoriasis', 'Fungal infection', 'Impetigo', 'Chicken pox'],
            'endocrinologist': ['Hyperthyroidism', 'Hypothyroidism', 'Diabetes', 'Hypoglycemia'],
            'infectious disease specialist': ['AIDS', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Tuberculosis', 'Typhoid', 'Dengue', 'Malaria', 'Pneumonia'],
            'gastroenterologist': ['Chronic cholestasis', 'Jaundice', 'Peptic ulcer disease', 'GERD', 'Alcoholic hepatitis', 'Gastroenteritis'],
            'cardiologist': ['Hypertension', 'Heart attack'],
            'neurologist': ['Migraine', '(vertigo) Paroxysmal Positional Vertigo', 'Paralysis (brain hemorrhage)'],
            'rheumatologist': ['Arthritis', 'Osteoarthritis', 'Cervical spondylosis', 'Varicose veins', 'Bronchial Asthma'],
            'urologist': ['Urinary tract infection'],
            'allergist': ['Allergy', 'Common Cold'],
            'general physician': ['Drug Reaction']
        }
        
        primary_disease = str(predictions[0]['disease']).lower()
        recommended_doctor = 'general physician'
        
        for doctor, keywords in doctor_types.items():
            if any(keyword.lower() in primary_disease for keyword in keywords):
                recommended_doctor = doctor
                break
        
        # Prepare response
        response_data = {
            'success': True,
            'predictions': convert_numpy_types(predictions),
            'selected_symptoms': selected_symptoms,
            'recommended_doctor': recommended_doctor.title(),
            'total_confidence': float(round(sum([p['confidence'] for p in predictions]), 2))
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

# Keep your other routes unchanged
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
            'symptoms': matched_symptoms[:10]
        })
    except Exception as e:
        return jsonify({'symptoms': [], 'error': str(e)})

@app.route('/get_all_symptoms')
def get_all_symptoms():
    return jsonify({
        'symptoms': list(symptom_list) 
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)