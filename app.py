import streamlit as st
import pandas as pd
import joblib

# Load the trained regression model
model = joblib.load("random_forest.pkl")
scaler= joblib.load("scaler.pkl")
# Define all expected columns exactly as used in training
expected_columns = [
    'Year', 'Duration', 'Votes',
    'Genre_Action, Adventure',
    'Genre_Action, Adventure, Biography',
    'Genre_Action, Adventure, Comedy',
    'Genre_Action, Adventure, Crime',
    'Genre_Action, Adventure, Drama',
    'Genre_Action, Adventure, Family',
    'Genre_Action, Adventure, Fantasy',
    'Director_Christopher Nolan',
    'Director_Steven Spielberg',
    'Actor 1_Leonardo DiCaprio',
    'Actor 2_Tom Hanks',
    'Actor 3_Kate Winslet'
]

# Streamlit UI
st.title("🎬 Movie Rating Predictor")

year = st.number_input("Release Year", min_value=1900, max_value=2030, step=1)
duration = st.number_input("Duration (in minutes)", min_value=30, max_value=300, step=1)
votes = st.number_input("Number of Votes", min_value=0, step=100)

genre = st.selectbox("Genre", [
    "Action, Adventure",
    "Action, Adventure, Biography",
    "Action, Adventure, Comedy",
    "Action, Adventure, Crime",
    "Action, Adventure, Drama",
    "Action, Adventure, Family",
    "Action, Adventure, Fantasy"
])

director = st.selectbox("Director", [
    "Christopher Nolan", "Steven Spielberg"
])

actor1 = st.selectbox("Actor 1", ["Leonardo DiCaprio"])
actor2 = st.selectbox("Actor 2", ["Tom Hanks"])
actor3 = st.selectbox("Actor 3", ["Kate Winslet"])

if st.button("Predict Rating"):
    # Initialize all values with 0
    input_data = {col: 0 for col in expected_columns}

    # Set base features
    input_data['Year'] = year
    input_data['Duration'] = duration
    input_data['Votes'] = votes

    # Set encoded categorical features
    genre_col = f"Genre_{genre}"
    director_col = f"Director_{director}"
    actor1_col = f"Actor 1_{actor1}"
    actor2_col = f"Actor 2_{actor2}"
    actor3_col = f"Actor 3_{actor3}"

    for col in [genre_col, director_col, actor1_col, actor2_col, actor3_col]:
        if col in input_data:
            input_data[col] = 1

    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Rating: {round(prediction, 2)}")
