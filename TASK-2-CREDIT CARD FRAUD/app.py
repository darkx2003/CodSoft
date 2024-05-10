import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model from the .pkl file
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define Streamlit app title and input block
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is fraudulent:")

# Input block
cc_num = st.text_input("Credit Card Number")
amt = st.number_input("Amount")
lat = st.number_input("Latitude")
long = st.number_input("Longitude")
merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")

# Predict function
def predict_fraud(cc_num, amt, lat, long, merch_lat, merch_long):
    features = np.array([cc_num, amt, lat, long, merch_lat, merch_long], dtype=np.float64).reshape(1, -1)
    prediction = model.predict(features)
    return prediction

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # Get prediction
    prediction = predict_fraud(cc_num, amt, lat, long, merch_lat, merch_long)
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")