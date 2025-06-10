import streamlit as st
import pandas as pd
import joblib

# Load model and column structure
model = joblib.load("random_forest.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler=jobib.load("scaler.pkl")
st.title("🎬 Movie Rating Predictor")

# User Inputs
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
duration = st.number_input("Duration (in minutes)", min_value=30, max_value=300)
votes = st.number_input("Votes", min_value=0)

# Genre - example genres based on your data
genre_options = [
    "Action, Adventure", "Action, Adventure, Biography", "Action, Adventure, Comedy",
    "Action, Adventure, Crime", "Action, Adventure, Drama", "Action, Adventure, Family",
    "Action, Adventure, Fantasy"
]
genre = st.selectbox("Genre", genre_options)

# Create input dictionary
input_data = {
    'Year': year,
    'Duration': duration,
    'Votes': votes,
    f'Genre_{genre}': 1
}

# Fill missing columns with 0
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[model_columns]

# Predict
if st.button("Predict Rating"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Rating: {round(prediction, 2)}")
