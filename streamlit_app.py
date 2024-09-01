import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

model = joblib.load('lin_reg13.pkl')
full_pipeline = joblib.load('pipeline.joblib')

def predict_price(features):
    features_df = pd.DataFrame([features], columns=[
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity',
    ])
    
    # Transformar las features usando el pipeline
    transformed_features = full_pipeline.transform(features_df)
    prediction = model.predict(transformed_features)
    return prediction[0]

st.title("Housing Price Prediction")
st.write("### Input Data")
col1, col2 = st.columns(2)
# Inputs
longitude = col1.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
latitude = col1.number_input("Latitude", min_value=0.0, max_value=90.0, value=0.0)
housing_median_age = col2.number_input("Housing Median Age", min_value=0.0, max_value=100.0, value=0.0)
total_rooms = col2.number_input("Total Rooms", min_value=0.0, value=0.0)
total_bedrooms = col1.number_input("Total Bedrooms", min_value=0.0, value=0.0)
population = col1.number_input("Population", min_value=0.0, value=0.0)
households = col2.number_input("Households", min_value=0.0, value=0.0)
median_income = col2.number_input("Median Income", min_value=0.0, value=0.0)
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

features = [
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity
]

st.write("### Output Data")
if st.button("Predict"):
    prediction = predict_price(features)
    st.write(f"El precio estimado de la casa es: ${prediction:,.2f}")