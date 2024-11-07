import streamlit as st
import numpy as np
import joblib
import pandas as pd
import zipfile
import io

# Load your model
model = joblib.load("Mobiles.pkl")

# Set page configuration
st.set_page_config(page_title="Mobile Hunt", page_icon=":iphone:", layout="centered")

# CSS styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f8f9fa;
    }
    .container {
        max-width: 500px;
        padding: 20px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        text-align: center;
        color: #007bff;
    }
    label {
        color: #6c757d;
    }
    .stButton button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Main app
#st.markdown("<div class='container'>", unsafe_allow_html=True)

st.title("📱 Mobile Hunt")
st.info("This app builds a machine learning model!")
st.image("image.jpeg", use_column_width=True)

import streamlit as st
import pandas as pd
import zipfile
import io

# Path to the zip file
zip_file_path = r"C:\Users\Akshitha\OneDrive\New folder\ExamScore.Prediction\archive (3).zip"

# Extract and read the CSV from the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # List all files in the zip and find CSVs
    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
    
    if csv_files:
        # Read the first CSV file in the zip
        with zip_ref.open(csv_files[0]) as csv_file:
            df = pd.read_csv(csv_file)

        # Use Streamlit's expander for data visualization
        with st.expander("Data visualization"):
            st.write("Scatter plot of device_model vs gender")
            
            # Ensure columns exist before plotting
            if "device_model" in df.columns and "gender" in df.columns:
                st.scatter_chart(data=df, x="Device Model", y="Gender")
            else:
                st.error("The columns 'device_model' or 'gender' do not exist in the dataset.")
    else:
        st.error("No CSV file found in the zip file.")


user_id = st.number_input("Please enter your ID:", min_value=0)
device_model = st.selectbox(
    "Device Model",
    [0, 1, 2, 3, 4],
    format_func=lambda x: ["Xiaomi", "iPhone", "Google", "OnePlus", "Samsung "][x]
)
app_usage_time = st.number_input("App Usage Time (in hours):", min_value=0.0)
battery_drain = st.number_input("Battery Drain (%):", min_value=0.0)
screen_on_time = st.number_input("Screen On Time (in hours):", min_value=0.0)
number_of_apps_installed = st.number_input("Number of Apps Installed:", min_value=0)
data_usage = st.number_input("Data Usage (in MB):", min_value=0.0)
age = st.number_input("Age:", min_value=0)
user_behavior_class = st.number_input("User Behavior Class:", min_value=0)
operating_system = st.selectbox(
    "Operating System",
    [0, 1],
    format_func=lambda x: "Android" if x == 0 else "iOS"
)

# Prediction button
if st.button('Predict Gender'):
    features = np.array([[user_id, device_model, app_usage_time, screen_on_time, battery_drain,
                          number_of_apps_installed, data_usage, age, operating_system, user_behavior_class]])
    
    # Predict gender
    output = model.predict(features)
    gender = 'Male' if output == 1 else 'Female'
    
    # Display result
    st.markdown(f"<h3 style='text-align: center;'>Gender Prediction: {gender}</h3>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
