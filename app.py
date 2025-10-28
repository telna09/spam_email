import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load the saved model and vectorizer
with open("spam_classifier.pkl", "rb") as f:
    vectorizer, tfidf_transformer, model = pickle.load(f)

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("Spam Email Classifier")
st.write("Enter an email message to check if it's spam or ham.")

user_input = st.text_area("Enter your email content here:", "")

if st.button("Classify"):
    if user_input:
        cleaned_email = preprocess_text(user_input)
        email_vector = vectorizer.transform([cleaned_email])
        email_tfidf = tfidf_transformer.transform(email_vector)
        prediction = model.predict(email_tfidf.toarray())
        
        result = "Spam" if prediction == 1 else "Ham"
        st.subheader(f"The email is classified as: {result}")
    else:
        st.warning("Please enter an email message.")

