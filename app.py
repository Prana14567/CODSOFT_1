import streamlit as st
import joblib
import pandas as pd
import numpy as np


model = joblib.load("movie_rating_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("expected_columns.pkl") 

st.title("🎬 Movie Rating Predictor")


duration = st.number_input("Duration (in minutes)", min_value=30, max_value=300, step=1)
votes = st.number_input("Number of Votes", min_value=0)
year = st.number_input("Release Year", min_value=1900, max_value=2100)
genre = st.text_input("Genre (e.g., Action, Drama)")
director = st.text_input("Director Name")
actor1 = st.text_input("Actor 1")
actor2 = st.text_input("Actor 2")
actor3 = st.text_input("Actor 3")


if st.button("Predict Rating"):
    
    input_dict = {
        "Duration": [duration],
        "Votes": [votes],
        "Year": [year],
        "Genre": [genre],
        "Director": [director],
        "Actor 1": [actor1],
        "Actor 2": [actor2],
        "Actor 3": [actor3]
    }
    input_df = pd.DataFrame(input_dict)


   
    input_df = pd.get_dummies(input_df)

   
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]  # Reorder columns

    
    numeric_cols = ["Duration", "Votes", "Year"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict rating
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Movie Rating: {round(prediction, 2)} ⭐")
