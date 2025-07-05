import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest

def detect_faults(df):
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['temperature', 'humidity', 'co2', 'pressure']])
    return df

def show_dashboard(df): 
    if 'anomaly' not in df.columns:
        st.error("Anomaly column is missing! Please run the fault detection model first.")
        return

    df['anomaly'] = df['anomaly'].astype(str).str.lower()
    anomaly_conditions = df['anomaly'].isin(['-1', 'anomaly', 'true', 'yes'])
    normal_conditions = df['anomaly'].isin(['1', 'normal', 'false', 'no'])

    st.write("## Overview Metrics")
    total_sensors = df.shape[0]
    faulty_sensors = df[anomaly_conditions]
    normal_sensors = df[normal_conditions]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sensors", total_sensors)
    with col2:
        st.metric("Sensors with Anomalies", faulty_sensors.shape[0])
    with col3:
        st.metric("Normal Sensors", normal_sensors.shape[0])

    
    st.write("## Correlation Analysis")
    st.write("### Sensor Correlations Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.drop(columns=['sensor_id']).select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Sensor Correlations")
    st.pyplot(fig)


    st.write("## Distribution Analysis")
    tab1, tab2 = st.tabs(["Value Distributions", "Anomaly Distribution"])
    
    with tab1:
        st.write("### Sensor Value Distributions")
        cols = st.columns(2)
        for i, column in enumerate([col for col in df.columns if col not in ['sensor_id', 'anomaly']]):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[column], kde=True, ax=ax, color='skyblue', bins=20)
                ax.set_title(f"Distribution of {column}")
                st.pyplot(fig)
    
    with tab2:
        st.write("### Anomaly Detection Analysis")
        anomaly_count = df['anomaly'].value_counts().reset_index()
        anomaly_count.columns = ['Anomaly', 'Count']
        fig = px.pie(anomaly_count, names='Anomaly', values='Count', 
                     title="Anomaly Detection Distribution")
        st.plotly_chart(fig, use_container_width=True)


    st.write("## Relationship Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### CO2 vs Temperature")
        fig = px.scatter(df, x="co2", y="temperature", color="anomaly", 
                         title="CO2 vs Temperature by Anomaly Status")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write("### Pressure Distribution")
        fig = px.box(df, y='pressure', 
                     title="Pressure Levels")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🌲 Forest Sensor Fault Detection Dashboard")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = detect_faults(df)
        show_dashboard(df)

if __name__ == "__main__":
    main()
