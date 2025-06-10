import streamlit as st
import pandas as pd
import joblib


model = joblib.load("random_forest.pkl")

scaler=joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")  # <-- create this during training!

# App title
st.title("🎬 Movie Rating Prediction")

# Input fields
duration = st.number_input("Duration (minutes)", min_value=30, max_value=300)
votes = st.number_input("Number of Votes", min_value=0)
year = st.number_input("Release Year", min_value=1900, max_value=2025)
genre = st.text_input("Genre (e.g., Action, Comedy)")
director = st.text_input("Director Name")
actor1 = st.text_input("Actor 1")
actor2 = st.text_input("Actor 2")
actor3 = st.text_input("Actor 3")

# Prediction logic
if st.button("Predict Rating"):
    # Create input dataframe
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

    # One-hot encode input features
    input_df = pd.get_dummies(input_df)

    # Reindex to match training columns
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Make prediction
    rating_pred = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Movie Rating: {round(rating_pred, 2)}")
