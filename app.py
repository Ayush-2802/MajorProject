import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('machinery_data.csv')
data.fillna(method='ffill', inplace=True)

# Feature selection and normalization
features = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']
target_rul = 'RUL'
target_maintenance = 'maintenance'
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split data for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data[features], data[target_rul], test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data[features], data[target_maintenance], test_size=0.2, random_state=42)

# Train models
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

# Prediction function
def predict_maintenance(features):
    rul_pred = reg_model.predict([features])
    maint_pred = clf_model.predict([features])
    cluster_pred = kmeans.predict([features])
    return {
        'RUL Prediction': rul_pred[0],
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'Normal',
        'Anomaly Detection': 'Anomaly' if cluster_pred[0] == 1 else 'Normal'
    }

# Add custom CSS for rectangular tab highlighting and animation
st.markdown("""
    <style>
    /* Decrease padding/margins in the main container */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 95%;
    }
    
    div[data-testid="stHorizontalBlock"] > div:first-child {
        display: flex;
        justify-content: space-between;
    }
    div[data-testid="stHorizontalBlock"] > div:first-child > div {
        border: 2px solid transparent;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stHorizontalBlock"] > div:first-child > div[aria-selected="true"] {
        border-color: #1DB954; /* Highlight color */
        background-color: #1e1e1e;
        transform: scale(1.05);
    }
    div[data-testid="stHorizontalBlock"] > div:first-child > div:hover {
        border-color: #1DB954;
        background-color: #2a2a2a;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit Top Navigation Tabs
tabs = st.tabs(["Home", "Input Data", "Results", "Visualizations","Data"])

with tabs[0]:
    st.title("Welcome to the Predictive Maintenance Dashboard")
    st.markdown("""
    This application provides predictive maintenance insights for industrial machinery. 
    Use the navigation tabs to explore different sections of the app.
    """)

with tabs[1]:
    st.title("Input Features")
    st.markdown("Use the sliders to input the sensor readings and operational hours or generate random values.")

    if 'generated_values' not in st.session_state:
        st.session_state['generated_values'] = None

    if st.button('Generate Random Values'):
        sensor_1 = np.random.uniform(data['sensor_1'].min(), data['sensor_1'].max())
        sensor_2 = np.random.uniform(data['sensor_2'].min(), data['sensor_2'].max())
        sensor_3 = np.random.uniform(data['sensor_3'].min(), data['sensor_3'].max())
        operational_hours = np.random.uniform(data['operational_hours'].min(), data['operational_hours'].max())
        st.session_state['generated_values'] = [sensor_1, sensor_2, sensor_3, operational_hours]
        st.success("Random values generated successfully!")

    if st.session_state['generated_values'] is not None:
        st.write("**Generated Values:**")
        st.write(f"Sensor 1: {st.session_state['generated_values'][0]:.2f}")
        st.write(f"Sensor 2: {st.session_state['generated_values'][1]:.2f}")
        st.write(f"Sensor 3: {st.session_state['generated_values'][2]:.2f}")
        st.write(f"Operational Hours: {st.session_state['generated_values'][3]:.2f}")

        if st.button('Use Generated Values'):
            st.session_state['input_features'] = st.session_state['generated_values']
            st.success("Generated values have been used. Navigate to the Results page to see the predictions.")

    st.markdown("**Or manually input values:**")
    sensor_1 = st.slider('Sensor 1', float(data['sensor_1'].min()), float(data['sensor_1'].max()), float(data['sensor_1'].mean()))
    sensor_2 = st.slider('Sensor 2', float(data['sensor_2'].min()), float(data['sensor_2'].max()), float(data['sensor_2'].mean()))
    sensor_3 = st.slider('Sensor 3', float(data['sensor_3'].min()), float(data['sensor_3'].max()), float(data['sensor_3'].mean()))
    operational_hours = st.slider('Operational Hours', int(data['operational_hours'].min()), int(data['operational_hours'].max()), int(data['operational_hours'].mean()))

    if st.button('Submit'):
        st.session_state['input_features'] = [sensor_1, sensor_2, sensor_3, operational_hours]
        st.success("Input data submitted successfully! Navigate to the Results page to see the predictions.")

with tabs[2]:
    st.title("Prediction Results")
    if 'input_features' not in st.session_state:
        st.warning("Please input data first in the 'Input Data' section.")
    else:
        input_features = st.session_state['input_features']
        prediction = predict_maintenance(input_features)
        st.write(f"**Remaining Useful Life (RUL):** {prediction['RUL Prediction']:.2f} hours")
        st.write(f"**Maintenance Status:** {prediction['Maintenance Prediction']}")
        st.write(f"**Anomaly Detection:** {prediction['Anomaly Detection']}")
        if prediction['Maintenance Prediction'] == 'Needs Maintenance':
            st.error('⚠️ Maintenance is required!')
        if prediction['Anomaly Detection'] == 'Anomaly':
            st.warning('⚠️ Anomaly detected in sensor readings!')

with tabs[3]:
    st.title("Data Visualizations")

    # Histogram for sensor readings
    st.subheader("Histogram of Sensor Readings")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True)
    axs[0].set_title('Sensor 1')
    sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True)
    axs[1].set_title('Sensor 2')
    sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True)
    axs[2].set_title('Sensor 3')
    st.pyplot(fig)

    # Scatter plot for sensor readings vs operational hours
    st.subheader("Scatter Plot of Sensor Readings vs Operational Hours")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.5)
    axs[0].set_title('Operational Hours vs Sensor 1')
    axs[0].set_xlabel('Operational Hours')
    axs[0].set_ylabel('Sensor 1')
    axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.5)
    axs[1].set_title('Operational Hours vs Sensor 2')
    axs[1].set_xlabel('Operational Hours')
    axs[1].set_ylabel('Sensor 2')
    axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.5)
    axs[2].set_title('Operational Hours vs Sensor 3')
    axs[2].set_xlabel('Operational Hours')
    axs[2].set_ylabel('Sensor 3')
    st.pyplot(fig)

    # Line chart for RUL over time
    st.subheader("Line Chart of RUL Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-')
    ax.set_title('RUL Over Operational Hours')
    ax.set_xlabel('Operational Hours')
    ax.set_ylabel('RUL')
    st.pyplot(fig)

    if 'input_features' in st.session_state:
        input_features = st.session_state['input_features']

        # Overlay generated input values if available
        if input_features is not None:
            # Histogram for sensor readings with generated input
            st.subheader("Histogram of Sensor Readings with Generated Input")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True)
            axs[0].set_title('Sensor 1')
            axs[0].axvline(input_features[0], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True)
            axs[1].set_title('Sensor 2')
            axs[1].axvline(input_features[1], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True)
            axs[2].set_title('Sensor 3')
            axs[2].axvline(input_features[2], color='red', linestyle='--', label='Generated Value')
            plt.legend()
            st.pyplot(fig)

            # Scatter plot for sensor readings vs operational hours with generated input
            st.subheader("Scatter Plot of Sensor Readings vs Operational Hours with Generated Input")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.5)
            axs[0].set_title('Operational Hours vs Sensor 1')
            axs[0].set_xlabel('Operational Hours')
            axs[0].set_ylabel('Sensor 1')
            axs[0].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            axs[0].legend()
            axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.5)
            axs[1].set_title('Operational Hours vs Sensor 2')
            axs[1].set_xlabel('Operational Hours')
            axs[1].set_ylabel('Sensor 2')
            axs[1].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            axs[1].legend()
            axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.5)
            axs[2].set_title('Operational Hours vs Sensor 3')
            axs[2].set_xlabel('Operational Hours')
            axs[2].set_ylabel('Sensor 3')
            axs[2].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            axs[2].legend()
            st.pyplot(fig)

            # Line chart for RUL over time with generated input
            st.subheader("Line Chart of RUL Over Time with Generated Input")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-')
            ax.set_title('RUL Over Operational Hours')
            ax.set_xlabel('Operational Hours')
            ax.set_ylabel('RUL')
            ax.axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            ax.legend()
            st.pyplot(fig)

with tabs[4]:
    st.title("Data")
    st.write(data.head(10))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json

app = Flask(__name__)
CORS(app)

# Load your trained model
# Replace this with the actual path to your model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to get predictions from the model
    """
    data = request.json
    try:
        # Preprocess the input data
        # Modify this according to your model's requirements
        input_data = pd.DataFrame(data['inputs'])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/data', methods=['GET'])
def get_data():
    """
    Endpoint to get visualization data
    """
    try:
        # Replace this with your actual data loading logic
        # Example: loading a CSV file
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
        df = pd.read_csv(data_path)
        
        # Basic statistics for dashboard
        stats = {
            'count': len(df),
            'features': df.columns.tolist(),
            'summary': json.loads(df.describe().to_json())
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'sample_data': json.loads(df.head(100).to_json(orient='records'))
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)