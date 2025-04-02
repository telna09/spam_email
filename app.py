import streamlit as st
import pickle
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    text = re.sub("\s+", " ", text).strip()  # Remove extra spaces
    return text

# Load models and vectorizers
with open('spam_classifierNB.pkl', 'rb') as nb_model_file:
    nb_classifier = pickle.load(nb_model_file)
with open('tfidf_vectorizerNB.pkl', 'rb') as nb_vectorizer_file:
    nb_vectorizer = pickle.load(nb_vectorizer_file)

with open('spam_classifierLR.pkl', 'rb') as lr_model_file:
    lr_classifier = pickle.load(lr_model_file)
with open('tfidf_vectorizerLR.pkl', 'rb') as lr_vectorizer_file:
    lr_vectorizer = pickle.load(lr_vectorizer_file)

def predict_spam(email_text, model_type):
    email_text_cleaned = clean_text(email_text)
    
    if model_type == "Naïve Bayes":
        email_tfidf = nb_vectorizer.transform([email_text_cleaned])
        prediction = nb_classifier.predict(email_tfidf)[0]
    else:
        email_tfidf = lr_vectorizer.transform([email_text_cleaned])
        prediction = lr_classifier.predict(email_tfidf)[0]
    
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit App
st.title("Spam Detector App")
st.write("Enter an email text below and select a classifier to predict whether it is Spam or Not Spam.")

# User input
user_input = st.text_area("Enter email text:")

# Model selection
model_type = st.radio("Choose a classifier:", ("Naïve Bayes", "Logistic Regression"))

if st.button("Predict"):
    if user_input.strip():
        prediction = predict_spam(user_input, model_type)
        st.write(f"**Prediction:** {prediction}")
    else:
        st.write("Please enter some text to classify.")
