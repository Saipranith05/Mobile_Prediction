import streamlit as st
import numpy as np
import joblib

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
st.markdown("<div class='container'>", unsafe_allow_html=True)

st.title("📱 Mobile Hunt")
st.image("image.jpeg", use_column_width=True)

# Input fields with placeholders for better UX
user_id = st.number_input("Please enter your ID:", min_value=0)
device_model = st.selectbox(
    "Device Model",
    [0, 1, 2, 3, 4],
    format_func=lambda x: ["Xiaomi Mi 11", "iPhone 12", "Google Pixel 5", "OnePlus 9", "Samsung Galaxy S21"][x]
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
