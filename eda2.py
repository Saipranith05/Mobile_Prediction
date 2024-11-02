import streamlit as st
import joblib 
import numpy as np

model = joblib.load("Tips.pkl")

Total_Bill = st.number_input("Please enter your total_bill: ")
Sex = st.selectbox("sex(0:Male, 1:Female)", [0,1])
Smoker = st.selectbox('smoker(0:Yes, 1:NO)', [0,1])
Day = st.selectbox("day(0:Thur, 1:Fri, 2:Sat, 3:Sun)", [0,1,2,3])
Time = st.selectbox("time(0:Lunch, 1:Dinner)", [0,1])
Size = st.number_input("Please enter your size: ")

if st.button("predict"):
    features = np.array([[Total_Bill, Sex, Smoker, Day, Time, Size]])
    output = model.predict(features)
    st.write(f"The Prediction of Tip amount is ${output[0]:.2f}")