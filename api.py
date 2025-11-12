"""
Flask API for Solar Power Forecasting
Converted from Streamlit app for Vercel deployment
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for cached data
CACHED_MODEL = None
CACHED_DATA = None


def load_model():
    """Load trained model or create a demo model"""
    global CACHED_MODEL
    
    if CACHED_MODEL is not None:
        return CACHED_MODEL
    
    # Try to load local model files
    model_files = ['solar_model.pkl', 'final_solar_forecasting_model_new_features.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                file_size = os.path.getsize(model_file)
                if file_size == 0:
                    continue
                    
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                CACHED_MODEL = model_data
                return model_data
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                continue
    
    # If no model found, create a simple demo model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    print("Using demo mode with simplified predictions.")
    demo_model = RandomForestRegressor(n_estimators=10, random_state=42)
    demo_scaler = StandardScaler()
    
    CACHED_MODEL = {
        'model': demo_model,
        'scaler': demo_scaler,
        'feature_columns': ['SOURCE_KEY', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 
                           'IRRADIATION', 'AC_POWER_lag1', 'AC_POWER_lag2', 'AC_POWER_lag3',
                           'IRRADIATION_lag1', 'IRRADIATION_lag2', 'IRRADIATION_lag3',
                           'AMBIENT_TEMPERATURE_lag1', 'AMBIENT_TEMPERATURE_lag2', 'AMBIENT_TEMPERATURE_lag3',
                           'MODULE_TEMPERATURE_lag1', 'MODULE_TEMPERATURE_lag2', 'MODULE_TEMPERATURE_lag3',
                           'IRRADIATION_rolling_mean_3', 'AMBIENT_TEMPERATURE_rolling_mean_3', 
                           'MODULE_TEMPERATURE_rolling_mean_3', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
        'demo_mode': True
    }
    return CACHED_MODEL


def load_data():
    """Load dataset from local storage"""
    global CACHED_DATA
    
    if CACHED_DATA is not None:
        return CACHED_DATA
    
    try:
        df = pd.read_csv('final_combined_Data_CI.csv')
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        CACHED_DATA = df
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_features(data_dict):
    """Create features from input data"""
    features = {}
    
    # Basic features
    features['SOURCE_KEY'] = data_dict.get('SOURCE_KEY', 0)
    features['DC_POWER'] = data_dict.get('DC_POWER', data_dict['IRRADIATION'] * 0.75)
    features['IRRADIATION'] = data_dict['IRRADIATION']
    features['AMBIENT_TEMPERATURE'] = data_dict['AMBIENT_TEMPERATURE']
    features['MODULE_TEMPERATURE'] = data_dict['MODULE_TEMPERATURE']
    
    # Lag features
    for feature_name in ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']:
        for lag in range(1, 4):
            features[f'{feature_name}_lag{lag}'] = data_dict.get(f'{feature_name}_lag{lag}', 0)
    
    # Rolling mean features
    for feature_name in ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']:
        features[f'{feature_name}_rolling_mean_3'] = data_dict.get(f'{feature_name}_rolling_mean_3', data_dict[feature_name])
    
    # Fourier features
    hour = data_dict.get('hour', 12)
    day = data_dict.get('day', 15)
    
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['day_sin'] = np.sin(2 * np.pi * day / 31)
    features['day_cos'] = np.cos(2 * np.pi * day / 31)
    
    return features


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['irradiation', 'ambient_temp', 'module_temp', 'hour', 'day']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Load model
        model_data = load_model()
        if model_data is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        
        # Prepare input data
        input_data = {
            'SOURCE_KEY': 0,
            'IRRADIATION': float(data['irradiation']),
            'AMBIENT_TEMPERATURE': float(data['ambient_temp']),
            'MODULE_TEMPERATURE': float(data['module_temp']),
            'hour': int(data['hour']),
            'day': int(data['day']),
        }
        
        # Add lag and rolling features
        for feature_name in ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']:
            for lag in range(1, 4):
                if feature_name == 'AC_POWER':
                    input_data[f'{feature_name}_lag{lag}'] = 0
                else:
                    input_data[f'{feature_name}_lag{lag}'] = input_data[feature_name]
        
        for feature_name in ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']:
            input_data[f'{feature_name}_rolling_mean_3'] = input_data[feature_name]
        
        features = create_features(input_data)
        
        # Create feature DataFrame
        if 'feature_columns' in model_data:
            feature_columns = model_data['feature_columns']
            for col in feature_columns:
                if col not in features:
                    features[col] = 0
            feature_df = pd.DataFrame([features])[feature_columns]
        else:
            feature_df = pd.DataFrame([features])
        
        # Make prediction
        is_demo = model_data.get('demo_mode', False)
        
        if is_demo:
            # Demo formula-based prediction
            base_power = float(data['irradiation']) * 0.7
            temp_factor = 1 - (float(data['module_temp']) - 25) * 0.004
            prediction = base_power * max(0.1, temp_factor)
            prediction = max(0, min(prediction, 1000))
        else:
            if scaler is not None:
                feature_scaled = scaler.transform(feature_df)
                prediction = float(model.predict(feature_scaled)[0])
            else:
                prediction = float(model.predict(feature_df)[0])
        
        # Calculate metrics
        efficiency = (prediction / float(data['irradiation']) * 100) if float(data['irradiation']) > 0 else 0
        daily_estimate = prediction * 24
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'efficiency': round(efficiency, 2),
            'daily_estimate': round(daily_estimate, 2),
            'demo_mode': is_demo
        })
    
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data-analysis', methods=['GET'])
def data_analysis():
    """API endpoint for data analysis"""
    try:
        df = load_data()
        
        if df is None or len(df) == 0:
            return jsonify({
                'success': False,
                'message': 'No dataset available'
            })
        
        # Summary statistics
        stats = {
            'total_records': len(df),
            'date_range_days': (df['DATE_TIME'].max() - df['DATE_TIME'].min()).days,
            'avg_ac_power': round(float(df['AC_POWER'].mean()), 2),
            'max_ac_power': round(float(df['AC_POWER'].max()), 2),
            'min_ac_power': round(float(df['AC_POWER'].min()), 2),
            'avg_irradiation': round(float(df['IRRADIATION'].mean()), 2),
            'avg_ambient_temp': round(float(df['AMBIENT_TEMPERATURE'].mean()), 2),
        }
        
        # Correlation analysis
        corr_features = ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
        corr_matrix = df[corr_features].corr()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'correlations': corr_matrix.to_dict()
        })
    
    except Exception as e:
        print(f"Error in data_analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """API endpoint for model information"""
    return jsonify({
        'success': True,
        'model_name': 'Random Forest Regressor',
        'mae': 0.380,
        'rmse': 2.576,
        'r2_score': 1.0,
        'features': [
            'Solar Irradiation (W/m²)',
            'Ambient Temperature (°C)',
            'Module Temperature (°C)',
            'Hour of day (Fourier)',
            'Day of month (Fourier)',
            'Lag features (previous 3 time steps)',
            'Rolling mean features (3-period)'
        ]
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


# Export for Vercel
application = app

if __name__ == '__main__':
    app.run(debug=True, port=5000)
