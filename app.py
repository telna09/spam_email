import streamlit as st
import pickle
import re
import string

# Function to clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    text = re.sub("\s+", " ", text).strip()  # Remove extra spaces
    return text

# Load pickle files
with open("tfidf_vectorizerNB.pkl", "rb") as f:
    vectorizer_nb = pickle.load(f)

with open("tfidf_vectorizerLR.pkl", "rb") as f:
    vectorizer_lr = pickle.load(f)

with open("spam_classifierNB.pkl", "rb") as f:
    classifier_nb = pickle.load(f)

with open("spam_classifierLR.pkl", "rb") as f:
    classifier_lr = pickle.load(f)

# Function to predict spam using both models
def predict_spam(email_text, vectorizer, model):
    email_text_cleaned = clean_text(email_text)
    email_tfidf = vectorizer.transform([email_text_cleaned])
    prediction = model.predict(email_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit UI
st.title("📧 Spam Email Classifier")
st.subheader("Enter an email below to check if it's Spam or Not")

# User input
user_input = st.text_area("✍️ Type your email content here:", "")

if st.button("🔍 Predict"):
    if user_input.strip():
        # Predict using Naïve Bayes
        nb_result = predict_spam(user_input, vectorizer_nb, classifier_nb)

        # Predict using Logistic Regression
        lr_result = predict_spam(user_input, vectorizer_lr, classifier_lr)

        # Display results
        st.subheader("📌 Results:")
        st.write(f"**Naïve Bayes Prediction:** {nb_result}")
        st.write(f"**Logistic Regression Prediction:** {lr_result}")

    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("🚀 Built with **Streamlit** | 📜 Model: Naïve Bayes & Logistic Regression")

