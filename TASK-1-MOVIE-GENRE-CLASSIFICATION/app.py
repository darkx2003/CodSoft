import pickle
import streamlit as st

# Load the trained model and TF-IDF vectorizer from the .pkl file
with open('movie_genre_classification_model.pkl', 'rb') as file:
    model, tfidf_vectorizer = pickle.load(file)

# Streamlit app title and input blocks
st.title("Movie Genre Classification")

title = st.text_input("Enter the movie title:")
description = st.text_area("Enter the movie description:")

# Predict function
def predict_genre(title, description):
    features = tfidf_vectorizer.transform([description])
    prediction = model.predict(features)
    return prediction[0]

# Create a button to submit input and get prediction
submit = st.button("Predict Genre")

if submit:
    # Get prediction
    genre = predict_genre(title, description)
    st.write("Predicted Genre:", genre)
