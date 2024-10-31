import streamlit as st
import numpy as np
import joblib 

model = joblib.load("Mobiles.pkl")

st.title("Mobile Hunt")
st.image("image.jpeg")

user_id = st.number_input("Please enter your ID: ")
device_model = st.selectbox("Device Model(0::Xiaomi Mi 11, 1:iPhone 12, 2:Google Pixel 5, 3:OnePlus 9, 4:Samsung Galaxy S21)", [0,1,2,3,4])
app_usage_time = st.number_input("Please enter your App Usage Time: ")
battery_drain = st.number_input("Please enter your Battary Drain: ")
screen_on_time = st.number_input("Please enter your Screen On Time: ")
number_of_apps_installed = st.number_input("Please enter your number of apps installed: ")
data_usage = st.number_input("Please enter your Data Usage: ")
age = st.number_input("Please enter your Age:")
user_behavior_class = st.number_input("Please enter your User Begaviour Class: ")

operating_system = st.selectbox("Operating System(0:Android, 1:ios)", [0,1])

if st.button('predict'):
    features = np.array([[user_id, device_model, app_usage_time, screen_on_time, battery_drain,
                         number_of_apps_installed, data_usage, age, operating_system, user_behavior_class ]])
    
    output = model.predict(features)
    st.write(f"Gender Prediction: {'Male' if output == 1 else 'Female'}")