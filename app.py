import streamlit as st
import joblib
import re
# Removed all imports related to NLTK (nltk, DownloadError, stopwords, word_tokenize)

# --- Model Loading Section ---
# NOTE: This solution assumes your original TF-IDF vectorizer ('tfidf_vectorizer.pkl')
# was configured with 'stop_words="english"' and handled tokenization internally.
# If your original model required NLTK's specific tokenization, the model may need
# to be retrained using scikit-learn's built-in tokenization/stop words.
try:
    model = joblib.load('trained_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.sidebar.success(
        "Model files loaded successfully. NLTK dependency removed.")
except FileNotFoundError:
    st.error("Model files ('trained_model.pkl' or 'tfidf_vectorizer.pkl') not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# --- Preprocessing Function (Simplified using only standard libraries) ---
def preprocess_text(text):
    """
    Cleans the input text using only standard Python libraries (re).
    Tokenization and stop word removal are now handled by the loaded TF-IDF vectorizer.
    """
    # Handle missing values
    text = text if isinstance(text, str) else ''
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation (keeping only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Streamlit UI ---

st.header("Cyberbullying Detection Framework (NLTK-Free)", divider='blue')
st.markdown("A simple model to classify text as potential cyberbullying or not.")

# Use a container for input and button for better layout
with st.container(border=True):
    user_input = st.text_area(
        "Enter a comment to check its classification:",
        placeholder="e.g., You are the worst person I have ever met. Go away.",
        height=150
    )

    # Use a styled button
    analyze_button = st.button(
        "Analyze Comment",
        type="primary",
        use_container_width=True
    )


# --- Analysis Logic ---
if analyze_button:
    if user_input:
        with st.spinner("Analyzing..."):
            # 1. Preprocess the user input
            preprocessed_input = preprocess_text(user_input)

            # 2. Vectorize the preprocessed input
            # Use the loaded TF-IDF vectorizer. transform requires an iterable (list of one string)
            # This step now handles tokenization and stop word removal automatically.
            input_vectorized = tfidf_vectorizer.transform([preprocessed_input])

            # 3. Make prediction (Assuming 1 is Cyberbullying, 0 is Not Cyberbullying)
            prediction = model.predict(input_vectorized)
            prediction_proba = model.predict_proba(
                input_vectorized) if hasattr(model, 'predict_proba') else None

        st.subheader("Analysis Result")

        if prediction[0] == 1:
            st.error("⚠️ **Classification: Cyberbullying**")
            st.balloons()
        else:
            st.success("✅ **Classification: Not Cyberbullying**")

        if prediction_proba is not None:
            col1, col2 = st.columns(2)
            prob_bully = prediction_proba[0][1]
            prob_safe = prediction_proba[0][0]

            with col1:
                st.metric("Bullying Probability", f"{prob_bully:.2%}")
            with col2:
                st.metric("Non-Bullying Probability", f"{prob_safe:.2%}")

    else:
        st.warning(
            "Please enter a comment in the text box before clicking 'Analyze'.")
