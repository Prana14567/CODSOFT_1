import streamlit as st
import pandas as pd
import joblib


model = joblib.load("random_forest.pkl")
scaler=joblib.load("scaler.pkl")

expected_columns = [
    'Year', 'Duration', 'Votes',
    'Genre_Action, Adventure',
    'Genre_Action, Adventure, Biography',
    'Genre_Action, Adventure, Comedy',
    'Genre_Action, Adventure, Crime',
    'Genre_Action, Adventure, Drama',
    'Genre_Action, Adventure, Family',
    'Genre_Action, Adventure, Fantasy',
    
    'Director_Steven Spielberg',
    'Director_Christopher Nolan',
    'Actor 1_Leonardo DiCaprio',
    'Actor 2_Tom Hanks',
    'Actor 3_Kate Winslet'
    # Add more actor/director columns based on your training
]

st.title("🎬 Movie Rating Predictor")

# Input widgets
year = st.number_input("Release Year", min_value=1900, max_value=2025)
duration = st.number_input("Movie Duration (in minutes)", min_value=30, max_value=300)
votes = st.number_input("Number of Votes", min_value=0)

genre = st.selectbox("Select Genre", [
    "Action, Adventure",
    "Action, Adventure, Biography",
    "Action, Adventure, Comedy",
    "Action, Adventure, Crime",
    "Action, Adventure, Drama",
    "Action, Adventure, Family",
    "Action, Adventure, Fantasy"
])

director = st.selectbox("Director", [
    "Steven Spielberg",
    "Christopher Nolan"
    # Add more based on training
])

actor1 = st.selectbox("Actor 1", [
    "Leonardo DiCaprio",
    # Add more based on training
])

actor2 = st.selectbox("Actor 2", [
    "Tom Hanks",
    # Add more
])

actor3 = st.selectbox("Actor 3", [
    "Kate Winslet",
    # Add more
])

# Predict Button
if st.button("Predict Rating"):
    # Create base dict
    input_dict = {
        "Year": [year],
        "Duration": [duration],
        "Votes": [votes]
    }

    # Create dummy variables for categorical features
    for col in expected_columns:
        if col not in input_dict:
            input_dict[col] = [0]

    # Activate matching genre
    genre_col = f"Genre_{genre}"
    if genre_col in expected_columns:
        input_dict[genre_col] = [1]

    # Activate director
    director_col = f"Director_{director}"
    if director_col in expected_columns:
        input_dict[director_col] = [1]

    # Actors
    actor1_col = f"Actor 1_{actor1}"
    if actor1_col in expected_columns:
        input_dict[actor1_col] = [1]

    actor2_col = f"Actor 2_{actor2}"
    if actor2_col in expected_columns:
        input_dict[actor2_col] = [1]

    actor3_col = f"Actor 3_{actor3}"
    if actor3_col in expected_columns:
        input_dict[actor3_col] = [1]

    
    input_df = pd.DataFrame(input_dict)

    
    input_df = input_df[expected_columns]


    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Rating: {round(prediction, 2)}")
