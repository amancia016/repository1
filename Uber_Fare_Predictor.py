
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from haversine import haversine
from datetime import datetime
import pydeck as pdk

# Load the trained model
model = joblib.load("uber_fare_stacking_ensemble.pkl")

st.title("🚖 Uber Fare Predictor")
st.markdown("Enter pickup and dropoff details to estimate fare.")

# Map input for pickup and dropoff
pickup = st.text_input("Pickup location (lat, lon)", "40.7614327, -73.9798156")
dropoff = st.text_input("Dropoff location (lat, lon)", "40.6513111, -73.8803331")

# Convert input to lat and lon
try:
    pickup_lat, pickup_lon = map(float, pickup.split(","))
    dropoff_lat, dropoff_lon = map(float, dropoff.split(","))
except:
    st.error("Please enter valid coordinates as 'lat, lon'.")
    st.stop()

pickup_point = [pickup_lat, pickup_lon]
dropoff_point = [dropoff_lat, dropoff_lon]

# Show map
layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame([
        {"lat": pickup_lat, "lon": pickup_lon},
        {"lat": dropoff_lat, "lon": dropoff_lon}
    ]),
    get_position='[lon, lat]',
    get_color='[0, 100, 255, 160]',
    get_radius=120,
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/streets-v12',
    initial_view_state=pdk.ViewState(
        latitude=(pickup_lat + dropoff_lat) / 2,
        longitude=(pickup_lon + dropoff_lon) / 2,
        zoom=11,
        pitch=45,
    ),
    layers=[layer],
))

# Passenger count input
passenger_count = st.slider("Number of passengers", 1, 6, 1)

# Date & Time input
ride_datetime = st.datetime_input("Select ride date and time", datetime.now())

# Calculate distance and other features
def build_features(pickup, dropoff, passenger_count, dt):
    distance_km = haversine(pickup, dropoff)
    return pd.DataFrame({
        "pickup_longitude": [pickup[1]],
        "pickup_latitude": [pickup[0]],
        "dropoff_longitude": [dropoff[1]],
        "dropoff_latitude": [dropoff[0]],
        "passenger_count": [passenger_count],
        "fare_distance": [distance_km],
        "hour": [dt.hour],
        "day": [dt.day],
        "month": [dt.month],
        "day_of_week": [dt.weekday()],
    })

features = build_features(pickup_point, dropoff_point, passenger_count, ride_datetime)

# Predict the fare
if st.button("💰 Predict Fare"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Fare: **${prediction:.2f}**")
