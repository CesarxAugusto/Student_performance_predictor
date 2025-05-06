import joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load("exam_score_predictor.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Grade Prediction")
st.header("Enter Student Information")

study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, step=1.0, format="%.1f")
mental_health = st.number_input("Mental health (scale 0 to 10)", min_value=0.0, max_value=10.0, step=1.0, format="%.1f")
exercise_frequency = st.number_input("Exercise frequency per week", min_value=0, max_value=7, step=1, format="%d")
sleep_hours = st.number_input("Sleep hours per day", min_value=0, max_value=24, step=1, format="%d")
attendance_percentage = st.number_input("Attendance percentage (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
social_media_time = st.number_input("Social media time per day (hours)", min_value=0, max_value=24, step=1, format="%d")
netflix_time = st.number_input("Netflix time per day (hours)", min_value=0, max_value=24, step=1, format="%d")
total_screen_time = social_media_time + netflix_time

input_dict = {
    'study_hours_per_day': [study_hours],
    'mental_health_rating': [mental_health],
    'exercise_frequency': [exercise_frequency],
    'sleep_hours': [sleep_hours],
    'attendance_percentage': [attendance_percentage],
    'total_screen_time': [total_screen_time]
}
input_df = pd.DataFrame(input_dict)
input_scaled = scaler.transform(input_df)

if st.button("Predict Grade"):
    prediction = model.predict(input_scaled)
    st.write(f"Prediction: {prediction}")