import streamlit as st
import pickle
import numpy as np

# Load the trained model & scaler
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("📉 Customer Churn Prediction App")
st.write("Enter customer details to predict if they will churn or not.")

# Input fields
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=75.5)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=900.0)

# Predict button
if st.button("Predict Churn"):
    # Prepare input data
    input_data = np.array([[tenure, monthly_charges, total_charges]])
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Churn" if prediction == 1 else "No Churn"

    # Display result
    st.success(f"Prediction: {result}")
