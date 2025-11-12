# Solar Power Forecasting with Streamlit

This is a solar power forecasting application built with Streamlit and Random Forest Regressor.

## Features

- **Real-time Predictions**: Predict AC power output based on environmental parameters
- **Interactive Dashboard**: Visualize predictions and historical data
- **Model Information**: View model performance metrics and details

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Model Performance

- **MAE**: 0.380 kW
- **RMSE**: 2.576 kW
- **R² Score**: 1.000

## Features Used

The model uses advanced feature engineering including:
- Environmental parameters (Irradiation, Temperature)
- Lag features
- Rolling mean features
- Fourier transformations for temporal patterns
