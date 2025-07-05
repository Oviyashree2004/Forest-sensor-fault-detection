import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from preprocess import preprocess_data
from dashboard import show_dashboard  

st.title("🌲 Forest Sensor Fault Detection Dashboard")

if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'detected_faults' not in st.session_state:
    st.session_state['detected_faults'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'upload'

if st.session_state['page'] == 'upload':
    uploaded_file = st.file_uploader("📂 Upload Sensor Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data")
        st.dataframe(df.head())
        df_clean = preprocess_data(df)
        features = df_clean.drop(columns=['sensor_id', 'fault_type'], errors='ignore')
        fault_detector = IsolationForest(contamination=0.1, random_state=42)
        df_clean['anomaly'] = fault_detector.fit_predict(features)
        df_clean['anomaly'] = df_clean['anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

        if 'anomaly' not in df_clean.columns:
            st.warning("⚠️ 'anomaly' column not found. Ensure the detection model ran successfully.")
            st.stop()

        st.session_state['df_clean'] = df_clean
        detected_faults = df_clean[df_clean['anomaly'] == 'Anomaly'].copy()

        def categorize_fault(row):
            if row.get('temperature', 0) > 45:
                return 'Overheating'
            elif row.get('humidity', 100) < 15:
                return 'Dry Fault'
            elif row.get('vibration', 0) > 10:
                return 'High Vibration'
            else:
                return 'General Fault'

        detected_faults['predicted_fault_type'] = detected_faults.apply(categorize_fault, axis=1)
        detected_faults = detected_faults.loc[:, ~detected_faults.columns.duplicated()]
        st.session_state['detected_faults'] = detected_faults

        st.success("✅ Dataset successfully processed.")
        if st.button("Go to Results"):
            st.session_state['page'] = "results"
            st.experimental_rerun()

if st.session_state['page'] == 'results':
    if st.session_state['df_clean'] is None:
        st.error("⚠️ No dataset found. Please upload a dataset first.")
        st.stop()
    if st.session_state['detected_faults'] is None:
        st.error("⚠️ No fault data found. Please run fault detection first.")
        st.stop()

    df_clean = st.session_state['df_clean']
    detected_faults = st.session_state['detected_faults']

    st.subheader("📋 Detected Faults with Predicted Type")
    st.dataframe(detected_faults)

    st.markdown("---")
    
    st.subheader("📊 Fault Dashboard ")
    show_dashboard(df_clean)
    st.subheader("✅ Final Recommendation")
    st.success("Based on the anomalies detected, please schedule maintenance for the affected sensors.")
    total_sensors = df_clean['sensor_id'].nunique() if 'sensor_id' in df_clean.columns else len(df_clean)
    faulty_sensors = detected_faults['sensor_id'].nunique() if 'sensor_id' in detected_faults.columns else len(detected_faults)
    anomalies_detected = len(detected_faults)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sensors Analyzed", total_sensors)
    col2.metric("Sensors with Faults", faulty_sensors)
    col3.metric("Total Fault Records", anomalies_detected)

