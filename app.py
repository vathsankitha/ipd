import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


# Load the trained model and TF-IDF vectorizer
model = joblib.load('trained_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define preprocessing function
def preprocess_text(text):
    # Handle missing values
    text = text if isinstance(text, str) else ''
    # Tokenization and remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    preprocessed_words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(preprocessed_words)

# Streamlit app title and description
st.title("Cyberbullying Detection Framework")
st.write("Enter a comment to check if it is cyberbullying.")

# Text input from user
user_input = st.text_area("Enter comment here:")

if st.button("Analyze"):
    if user_input:
        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)

        # Vectorize the preprocessed input using the loaded TF-IDF vectorizer
        input_vectorized = tfidf_vectorizer.transform([preprocessed_input]).toarray()

        # Make prediction
        prediction = model.predict(input_vectorized)

        # Display the result
        if prediction[0] == 1:
            st.error("This comment is likely cyberbullying.")
        else:
            st.success("This comment is not likely cyberbullying.")
    else:
        st.warning("Please enter a comment to analyze.")
