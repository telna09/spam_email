import streamlit as st
import pickle
import re
import string

# Function to load pickle files
def load_pickle(filename):
    """Loads a pickle file."""
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: File {filename} not found.")
        return None
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# Function to clean input text
def clean_text(text):
    """Preprocesses the input email text by removing punctuation, numbers, and extra spaces."""
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = text.lower()
    text = re.sub(r"[{}]".format(string.punctuation), "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Load the pickled models and vectorizers
tfidf_vectorizerNB = load_pickle("tfidf_vectorizerNB.pkl")
tfidf_vectorizerLR = load_pickle("tfidf_vectorizerLR.pkl")
spam_classifierNB = load_pickle("spam_classifierNB.pkl")
spam_classifierLR = load_pickle("spam_classifierLR.pkl")

# Ensure models are loaded successfully
if None in [tfidf_vectorizerNB, tfidf_vectorizerLR, spam_classifierNB, spam_classifierLR]:
    st.error("One or more model files could not be loaded. Check file paths.")

# Function to predict spam
def predict_spam(email_text, vectorizer, model):
    """Predicts whether an email is spam or not using the given model and vectorizer."""
    email_text_cleaned = clean_text(email_text)
    email_tfidf = vectorizer.transform([email_text_cleaned])
    prediction = model.predict(email_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit UI
st.title("📧 Spam Email Classifier 🚀")
st.write("Enter an email text below to check if it's Spam or Not Spam.")

user_input = st.text_area("Enter email content here:")

if st.button("Predict using Naïve Bayes"):
    if tfidf_vectorizerNB and spam_classifierNB:
        nb_result = predict_spam(user_input, tfidf_vectorizerNB, spam_classifierNB)
        st.write(f"**Naïve Bayes Prediction:** {nb_result}")
    else:
        st.error("Naïve Bayes model is not loaded properly.")

if st.button("Predict using Logistic Regression"):
    if tfidf_vectorizerLR and spam_classifierLR:
        lr_result = predict_spam(user_input, tfidf_vectorizerLR, spam_classifierLR)
        st.write(f"**Logistic Regression Prediction:** {lr_result}")
    else:
        st.error("Logistic Regression model is not loaded properly.")
