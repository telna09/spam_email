import streamlit as st
import pickle
import re
import string

def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Load the pre-trained models and vectorizer
vectorizer = load_pickle("tfidf_vectorizerNBCl.pkl")
nb_classifier = load_pickle("NBCl.pkl")
lr_classifier = load_pickle("spam_classifier.pkl")

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    text = re.sub("\s+", " ", text).strip()  # Remove extra spaces
    return text

# Function to predict spam
def predict_spam(email_text, model):
    email_text_cleaned = clean_text(email_text)
    email_tfidf = vectorizer.transform([email_text_cleaned])
    prediction = model.predict(email_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit UI
st.title("Spam Email Detector")
st.write("Enter an email message below to check if it is spam or not.")

# User input
user_input = st.text_area("Email Text:")

if st.button("Check for Spam"):
    if user_input:
        nb_result = predict_spam(user_input, nb_classifier)
        lr_result = predict_spam(user_input, lr_classifier)
        
        st.subheader("Results:")
        st.write(f"**Naïve Bayes Prediction:** {nb_result}")
        st.write(f"**Logistic Regression Prediction:** {lr_result}")
    else:
        st.warning("Please enter an email text to analyze.")

st.write("This app uses Machine Learning models to detect spam emails.")