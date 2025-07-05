from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def detect_faults(df):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    features = df.drop(columns=['sensor_id'])
    df['anomaly'] = model.fit_predict(features)
    df['anomaly'] = df['anomaly'].map({-1: 'Anomaly', 1: 'Normal'})
    df['fault_type'] = 'Healthy'
    anomaly_rows = df[df['anomaly'] == 'Anomaly']
    for index, row in anomaly_rows.iterrows():
        fault_scores = {
            'Temperature': abs(row['temperature'] - features['temperature'].mean()),
            'Humidity': abs(row['humidity'] - features['humidity'].mean()),
            'CO2': abs(row['co2'] - features['co2'].mean()),
            'Pressure': abs(row['pressure'] - features['pressure'].mean())
        }
        fault_type = max(fault_scores, key=fault_scores.get) + " Fault"
        df.at[index, 'fault_type'] = fault_type
    fault_count = df[df['anomaly'] == 'Anomaly'].shape[0]
    return df, fault_count
