from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from datetime import datetime
import sys
import os
# Add parent directory to python path so 'python src/app.py' works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import create_features

app = Flask(__name__)

# Load model and feature names
try:
    model_data = joblib.load('models/xgboost_production.pkl')
    model = model_data['model']
    feature_names = model_data['features']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model. Ensure model is trained. {e}")
    model = None
    feature_names = []

# In-memory store for recent readings per machine to calculate rolling features
# In production, use Redis or a time-series database.
# Dictionary mapping machine_id -> list of readings
history = defaultdict(list)
MAX_HISTORY = 24 # We need at least 12 for the 12h rolling window

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    
    if not data or 'machine_id' not in data or 'readings' not in data:
        return jsonify({"error": "Invalid payload format"}), 400
        
    machine_id = data['machine_id']
    readings = data['readings'] # Expected to be a dict of current readings: {'temperature': 65.5, 'vibration': 2.1, 'pressure': 110.0}
    timestamp = data.get('timestamp', datetime.now().isoformat())
    
    # Update history
    reading_entry = {
        'timestamp': pd.to_datetime(timestamp),
        'machine_id': machine_id,
        **readings
    }
    
    history[machine_id].append(reading_entry)
    
    # Keep only the latest MAX_HISTORY records
    if len(history[machine_id]) > MAX_HISTORY:
        history[machine_id] = history[machine_id][-MAX_HISTORY:]
        
    # We need at least enough history to generate features (though create_features uses min_periods=1)
    df_history = pd.DataFrame(history[machine_id])
    
    # Generate features
    # Since create_features expects a dataframe with multiple rows potentially,
    # and groups by machine_id, it will work on this single machine's history.
    try:
        df_features = create_features(df_history)
        
        # We only want to predict for the latest reading
        latest_features = df_features.iloc[-1:]
        
        # Select only the features the model was trained on
        X = latest_features[feature_names]
        
        # Predict probability
        prob = model.predict_proba(X)[0, 1]
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            "machine_id": machine_id,
            "failure_probability": float(prob),
            "alert": bool(prob > 0.5), # Configurable threshold
            "response_time_ms": round(response_time_ms, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
