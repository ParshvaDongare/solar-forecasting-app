import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import urllib.request

# Page configuration
st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        color: #FF6B35;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to download file from URL
def download_file(url, filename):
    """Download file from URL if not exists locally"""
    if not os.path.exists(filename):
        try:
            st.info(f"⬇️ Downloading {filename} from cloud...")
            import urllib.request
            import urllib.error
            
            # Create request with headers to avoid redirects
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            with urllib.request.urlopen(req) as response:
                # Check if we got redirected to HTML page
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    st.error(f"❌ Got HTML page instead of file (Content-Type: {content_type})")
                    st.info("This usually means GitHub is redirecting to a download page")
                    return False
                
                # Download the file
                with open(filename, 'wb') as f:
                    f.write(response.read())
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
                st.success(f"✅ Downloaded {filename} ({file_size:.1f} MB)")
                return True
            else:
                st.error(f"❌ File {filename} not created after download")
                return False
        except urllib.error.HTTPError as e:
            st.error(f"❌ HTTP Error downloading {filename}: {e.code} {e.reason}")
            return False
        except Exception as e:
            st.error(f"❌ Download failed for {filename}: {str(e)}")
            return False
    else:
        st.info(f"✅ Using existing {filename}")
    return True

# Load the trained model
@st.cache_resource
def load_model():
    """Load trained model or create a demo model"""
    import os
    
    # Check for MODEL_URL in secrets (safe check)
    try:
        if 'MODEL_URL' in st.secrets:
            model_url = st.secrets['MODEL_URL']
            if download_file(model_url, 'solar_model.pkl'):
                try:
                    # Verify file isn't empty
                    file_size = os.path.getsize('solar_model.pkl')
                    if file_size == 0:
                        st.error("❌ Model file is empty")
                        raise ValueError("Empty model file")
                    
                    with open('solar_model.pkl', 'rb') as f:
                        model_data = pickle.load(f)
                    st.success(f"✅ Loaded model from solar_model.pkl ({file_size / (1024*1024):.1f} MB)")
                    return model_data
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.info("⚠️ Falling back to demo mode")
    except (FileNotFoundError, AttributeError):
        pass  # No secrets file, continue to local files
    
    # Try local model files
    model_files = ['solar_model.pkl', 'final_solar_forecasting_model_new_features.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                # Verify file isn't empty
                file_size = os.path.getsize(model_file)
                if file_size == 0:
                    st.warning(f"⚠️ {model_file} is empty, skipping")
                    continue
                    
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                st.success(f"✅ Loaded model from {model_file} ({file_size / (1024*1024):.1f} MB)")
                return model_data
            except Exception as e:
                st.warning(f"⚠️ Error loading {model_file}: {str(e)}")
                continue
    
    # If no model found, create a simple demo model
    st.warning("⚠️ Pre-trained model not found or corrupted. Using demo mode with simplified predictions.")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple demo model
    demo_model = RandomForestRegressor(n_estimators=10, random_state=42)
    demo_scaler = StandardScaler()
    
    return {
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

# Load data
@st.cache_data
def load_data():
    """Load dataset from cloud or local storage"""
    # Check for DATA_URL in secrets (safe check)
    try:
        if 'DATA_URL' in st.secrets:
            data_url = st.secrets['DATA_URL']
            if download_file(data_url, 'final_combined_Data_CI.csv'):
                try:
                    df = pd.read_csv('final_combined_Data_CI.csv')
                    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
                    st.success("✅ Loaded data from cloud storage")
                    return df
                except Exception as e:
                    st.warning(f"⚠️ Error loading cloud data: {e}")
    except (FileNotFoundError, AttributeError):
        pass  # No secrets file, continue to local files
    
    # Try local file
    try:
        df = pd.read_csv('final_combined_Data_CI.csv')
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        return df
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Data Analysis tab will be limited.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Feature engineering function
def create_features(data_dict):
    """Create features from input data"""
    features = {}
    
    # Basic features
    features['SOURCE_KEY'] = data_dict.get('SOURCE_KEY', 0)
    features['DC_POWER'] = data_dict.get('DC_POWER', data_dict['IRRADIATION'] * 0.75)  # Estimate from irradiation
    features['IRRADIATION'] = data_dict['IRRADIATION']
    features['AMBIENT_TEMPERATURE'] = data_dict['AMBIENT_TEMPERATURE']
    features['MODULE_TEMPERATURE'] = data_dict['MODULE_TEMPERATURE']
    
    # Lag features (using default values for demo)
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

def main():
    # Title and description
    st.title("☀️ Solar Power Forecasting System")
    st.markdown("### Predict AC Power Output using Machine Learning")
    st.markdown("---")
    
    # Load model and data
    model_data = load_model()
    df = load_data()
    
    if model_data is None or df is None:
        st.error("Failed to load model or data. Please check your files.")
        return
    
    # Extract model and scaler
    if isinstance(model_data, dict):
        model = model_data.get('model')
        scaler = model_data.get('scaler')
    else:
        model = model_data
        scaler = None
    
    # Sidebar
    st.sidebar.header("🎯 Prediction Settings")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Data Analysis", "ℹ️ Model Info"])
    
    with tab1:
        st.header("Make Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environmental Parameters")
            irradiation = st.slider("Solar Irradiation (W/m²)", 0.0, 1200.0, 600.0, 10.0)
            ambient_temp = st.slider("Ambient Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
            module_temp = st.slider("Module Temperature (°C)", -10.0, 70.0, 35.0, 0.5)
        
        with col2:
            st.subheader("Time Parameters")
            date_time = st.date_input("Date", datetime.now())
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = date_time.day
            
            st.info(f"Selected: {date_time} at {hour}:00")
        
        st.markdown("---")
        
        if st.button("🔮 Predict AC Power", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'SOURCE_KEY': 0,
                'IRRADIATION': irradiation,
                'AMBIENT_TEMPERATURE': ambient_temp,
                'MODULE_TEMPERATURE': module_temp,
                'hour': hour,
                'day': day,
            }
            
            # Add lag features (using current values as approximation)
            for feature_name in ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']:
                for lag in range(1, 4):
                    if feature_name == 'AC_POWER':
                        input_data[f'{feature_name}_lag{lag}'] = 0
                    else:
                        input_data[f'{feature_name}_lag{lag}'] = input_data[feature_name]
            
            # Add rolling mean features
            for feature_name in ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']:
                input_data[f'{feature_name}_rolling_mean_3'] = input_data[feature_name]
            
            features = create_features(input_data)
            
            # Create DataFrame with correct feature order matching the model
            if model_data and 'feature_columns' in model_data:
                feature_columns = model_data['feature_columns']
                # Ensure all required features are present
                for col in feature_columns:
                    if col not in features:
                        features[col] = 0  # Default value for missing features
                # Create DataFrame with columns in the correct order
                feature_df = pd.DataFrame([features])[feature_columns]
            else:
                feature_df = pd.DataFrame([features])
            
            # Make prediction
            try:
                # Check if we're in demo mode
                is_demo = model_data.get('demo_mode', False)
                
                if is_demo:
                    # Simple formula-based prediction for demo
                    # AC_Power is roughly proportional to irradiation and affected by temperature
                    base_power = irradiation * 0.7  # Rough conversion factor
                    temp_factor = 1 - (module_temp - 25) * 0.004  # Temperature derating
                    prediction = base_power * max(0.1, temp_factor)
                    prediction = max(0, min(prediction, 1000))  # Clamp between 0-1000 kW
                    st.info("🚧 Demo mode: Using simplified formula-based prediction")
                elif scaler is not None:
                    feature_scaled = scaler.transform(feature_df)
                    prediction = model.predict(feature_scaled)[0]
                else:
                    prediction = model.predict(feature_df)[0]
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted AC Power", f"{prediction:.2f} kW", 
                             delta=None)
                
                with col2:
                    efficiency = (prediction / irradiation * 100) if irradiation > 0 else 0
                    st.metric("Estimated Efficiency", f"{efficiency:.2f}%")
                
                with col3:
                    daily_estimate = prediction * 24
                    st.metric("Daily Estimate", f"{daily_estimate:.2f} kWh")
                
                # Visualization
                st.markdown("---")
                st.subheader("📊 Prediction Visualization")
                
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "AC Power (kW)"},
                    gauge={
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 300], 'color': "lightgray"},
                            {'range': [300, 600], 'color': "gray"},
                            {'range': [600, 1000], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 800
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.error("Please check that the model and input features are compatible.")
    
    with tab2:
        st.header("Historical Data Analysis")
        
        if df is not None and len(df) > 0:
            # Summary statistics
            st.subheader("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Date Range", f"{(df['DATE_TIME'].max() - df['DATE_TIME'].min()).days} days")
            with col3:
                st.metric("Avg AC Power", f"{df['AC_POWER'].mean():.2f} kW")
            with col4:
                st.metric("Max AC Power", f"{df['AC_POWER'].max():.2f} kW")
            
            st.markdown("---")
            
            # Time series plot
            st.subheader("🕐 AC Power Over Time")
            
            # Sample data for better performance
            df_sample = df.sample(min(10000, len(df))).sort_values('DATE_TIME')
            
            fig = px.line(df_sample, x='DATE_TIME', y='AC_POWER', 
                         title='AC Power Generation Over Time',
                         labels={'AC_POWER': 'AC Power (kW)', 'DATE_TIME': 'Date & Time'})
            fig.update_traces(line_color='green')
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Feature Correlations")
            
            corr_features = ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
            corr_matrix = df[corr_features].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto=True,
                           color_continuous_scale='RdYlGn',
                           aspect='auto',
                           title='Feature Correlation Heatmap')
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            st.subheader("📈 Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='AC_POWER', nbins=50,
                                  title='AC Power Distribution',
                                  labels={'AC_POWER': 'AC Power (kW)'})
                fig.update_traces(marker_color='green')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df, x='IRRADIATION', nbins=50,
                                  title='Irradiation Distribution',
                                  labels={'IRRADIATION': 'Irradiation (W/m²)'})
                fig.update_traces(marker_color='orange')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📊 No dataset available for analysis.")
            st.info("""To enable data analysis:
            1. Upload your dataset to cloud storage (Google Drive, Dropbox, S3)
            2. Add DATA_URL to Streamlit Cloud secrets with the download link
            3. Restart the app""")
    
    with tab3:
        st.header("Model Information")
        
        st.subheader("🏆 Best Performing Model: Random Forest Regressor")
        
        st.markdown("""
        ### Model Performance Metrics
        
        - **MAE (Mean Absolute Error)**: 0.380 kW
        - **RMSE (Root Mean Squared Error)**: 2.576 kW
        - **R² Score**: 1.000
        
        ### Features Used
        The model uses the following features for prediction:
        
        1. **Environmental Features**:
           - Solar Irradiation (W/m²)
           - Ambient Temperature (°C)
           - Module Temperature (°C)
        
        2. **Time-based Features**:
           - Hour of day (Fourier transformed)
           - Day of month (Fourier transformed)
        
        3. **Lag Features**:
           - Previous 3 time steps for all environmental variables
        
        4. **Rolling Mean Features**:
           - 3-period rolling average for environmental variables
        
        ### Model Architecture
        - **Algorithm**: Random Forest Regressor
        - **Number of Estimators**: 100
        - **Random State**: 42
        - **Feature Engineering**: Advanced time-series features with lag and rolling statistics
        
        ### Dataset
        - **Source**: Solar power generation data from renewable energy plants
        - **Size**: Multiple plants with continuous monitoring
        - **Features**: Environmental conditions and power output measurements
        """)
        
        st.info("💡 This model achieved the best performance after implementing advanced feature engineering techniques including lag features, rolling statistics, and Fourier transformations for temporal patterns.")

if __name__ == "__main__":
    main()
