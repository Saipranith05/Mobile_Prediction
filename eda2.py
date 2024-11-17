import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("Tips.pkl")

# Streamlit app
st.title("Tip Predictor")
st.image("download.jpeg")

# Input fields
Total_Bill = st.number_input("Please enter your total bill: ", min_value=0.0, step=0.01)
Sex = st.selectbox("Sex (0: Male, 1: Female)", ["Male", "Female"])
Smoker = st.selectbox("Smoker (0: Yes, 1: No)", [0, 1])
Day = st.selectbox("Day (0: Thur, 1: Fri, 2: Sat, 3: Sun)", [0, 1, 2, 3])
Time = st.selectbox("Time (0: Lunch, 1: Dinner)", [0, 1])
Size = st.number_input("Please enter your party size: ", min_value=1, step=1)

# Map categorical inputs to numeric values
Sex_map = {"Male": 0, "Female": 1}
Day_map = {"Thur": 0, "Fri":1, "Sat":2, "Sun":3}
Time_map = {"Lunch": 0, "Dinner": 1}
Smoker_map = {"Yes":0, "No":1}

Sex_numeric = Sex_map[Sex]
Day_numeric = Day_map[Smoker]
Time_numeric = Time_map[Time]
Smoker_numeric = Smoker_map[Smoker] 


# Prediction
if st.button("Predict"):
    # Prepare the feature array
    features = np.array([[Total_Bill, Sex_numeric, Smoker, Day, Time, Size]])
    
    # Make the prediction
    output = model.predict(features)
    
    # Display the prediction
    st.write(f"The Prediction of Tip Amount is ${output[0]:.2f}")
