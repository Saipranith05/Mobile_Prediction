import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("Exam_Score.pkl")

# Streamlit app
st.title("Exam Score Prediction")

# Input fields
Hours_Studited = st.number_input("Please enter no. of Hours Studied: ")
Attendance = st.number_input("Please enter your Attendance: ")
Sleep_Hours = st.number_input("Please enter your Sleep Hours: ")
Previous_Scores = st.number_input("Please enter your Previous Scores: ")
Family_Income = st.selectbox("Family Income (0: Low, 1: Medium, 2: High)", ["Low", "Medium", "High"])
School_Type = st.selectbox("School Type (0: Private, 1: Public)", ["Private", "Public"])
Gender = st.selectbox("Gender (0: Male, 1: Female)", ["Male", "Female"])

# Map categorical inputs to numeric values
Family_Income_map = {"Low": 0, "Medium": 1, "High": 2}
School_Type_map = {"Private": 0, "Public": 1}
Gender_map = {"Male": 0, "Female": 1}

Family_Income_numeric = Family_Income_map[Family_Income]
School_Type_numeric = School_Type_map[School_Type]
Gender_numeric = Gender_map[Gender]

# Predict button
if st.button('Predict'):
    # Prepare the feature array
    features = np.array([[Hours_Studited, Attendance, Sleep_Hours, Previous_Scores, Family_Income_numeric, School_Type_numeric, Gender_numeric]])
    
    # Make the prediction
    output = model.predict(features)
    
    # Display the prediction
    st.write(f"Prediction of Exam Score for the Student: {output[0]}")
