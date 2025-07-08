import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler (1).pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()

st.set_page_config(page_title="🏠 California House Price Predictor", layout="centered")

st.title("🏠 California House Price Predictor")
st.write("Enter property details below to estimate median house value.")

# User input section
def get_user_input():
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", value=-120.0)
        latitude = st.number_input("Latitude", value=35.0)
        housing_median_age = st.slider("Housing Median Age", 1, 100, 30)
        total_rooms = st.number_input("Total Rooms", value=2000)
        total_bedrooms = st.number_input("Total Bedrooms", value=400)
        population = st.number_input("Population", value=1000)

    with col2:
        households = st.number_input("Households", value=500)
        median_income = st.number_input("Median Income (in $10k)", value=3.0)
        ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

    # Derived features
    total_rooms = np.log(total_rooms + 1)
    total_bedrooms = np.log(total_bedrooms + 1)
    population = np.log(population + 1)
    households = np.log(households + 1)

    # Encode ocean proximity
    ocean_dict = {"<1H OCEAN": [1, 0, 0, 0, 0],
                  "INLAND": [0, 1, 0, 0, 0],
                  "ISLAND": [0, 0, 1, 0, 0],
                  "NEAR BAY": [0, 0, 0, 1, 0],
                  "NEAR OCEAN": [0, 0, 0, 0, 1]}

    ocean_encoded = ocean_dict[ocean_proximity]

    # Derived ratios
    bedroom_ratio = total_bedrooms / total_rooms
    household_rooms = total_rooms / households

    # Final feature array
    features = np.array([
        [longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
         population, households, median_income] + ocean_encoded +
        [bedroom_ratio, household_rooms]
    ])

    return features

features = get_user_input()

if st.button("Predict House Value"):
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    st.success(f"🏡 Predicted Median House Value: ${prediction:,.2f}")

 #display input summary
    st.subheader("Input Summary")
    st.write(pd.DataFrame(features, columns=[
        "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
        "population", "households", "median_income",
        "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN",
        "bedroom_ratio", "household_rooms"
    ]))
