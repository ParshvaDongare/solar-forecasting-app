import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("Loading data...")
df = pd.read_csv('final_combined_Data_CI.csv')

# Convert DATE_TIME and prepare data
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
df.drop(['PLANT_ID'], axis=1, inplace=True)

# Extract time features
df['hour'] = df['DATE_TIME'].dt.hour
df['day'] = df['DATE_TIME'].dt.day
df['month'] = df['DATE_TIME'].dt.month
df['weekday'] = df['DATE_TIME'].dt.weekday

# Encode SOURCE_KEY
le = LabelEncoder()
df['SOURCE_KEY'] = le.fit_transform(df['SOURCE_KEY'])

# Sort by SOURCE_KEY and DATE_TIME
df = df.sort_values(by=['SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)

# Feature engineering - lag features
features_to_transform = ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
for feature in features_to_transform:
    for lag in range(1, 4):
        df[f'{feature}_lag{lag}'] = df.groupby('SOURCE_KEY')[feature].shift(lag)

# Rolling mean features
rolling_features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
for feature in rolling_features:
    df[f'{feature}_rolling_mean_3'] = df.groupby('SOURCE_KEY')[feature].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# Fourier features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

# Fill NaN values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop DATE_TIME
df.drop(['DATE_TIME'], axis=1, inplace=True)

print("Preparing features...")
# Define features and target
X = df.drop(['AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 'hour', 'day', 'month', 'weekday'], axis=1)
y = df['AC_POWER']

print(f"Feature columns: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest model...")
# Train Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

print("Evaluating model...")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = rf_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Save model and scaler
print("\nSaving model...")
model_data = {
    'model': rf_model,
    'scaler': scaler,
    'feature_columns': list(X.columns)
}

with open('solar_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully to 'solar_model.pkl'")
