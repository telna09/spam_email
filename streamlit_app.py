import streamlit as st
import joblib
import string
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="wide"
)

# Text cleaning function (same as training)
def clean_and_correct(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        lr_model = joblib.load('logistic_regression_model.joblib')
        nb_model = joblib.load('naive_bayes_model.joblib')
        return vectorizer, lr_model, nb_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure all .joblib files are in the same directory as this app.")
        return None, None, None

# Prediction function with confidence
def predict_with_confidence(message, model_choice, vectorizer, lr_model, nb_model):
    # Clean the message
    message_clean = clean_and_correct(message)
    
    # Vectorize
    message_vec = vectorizer.transform([message_clean])
    
    # Choose model and predict
    if model_choice == "Logistic Regression":
        model = lr_model
        prediction = model.predict(message_vec)[0]
        # Get probability scores
        probabilities = model.predict_proba(message_vec)[0]
        confidence = max(probabilities) * 100
        
    else:  # Naive Bayes
        model = nb_model
        prediction = model.predict(message_vec)[0]
        # Get probability scores
        probabilities = model.predict_proba(message_vec)[0]
        confidence = max(probabilities) * 100
    
    # Convert prediction to label
    result = 'Spam' if prediction == 1 else 'Ham'
    
    return result, confidence, probabilities

# Main app
def main():
    # Title and description
    st.title("📧 Spam Detection App")
    st.markdown("---")
    st.markdown("### Detect whether a message is Spam or Ham using Machine Learning")
    
    # Load models
    vectorizer, lr_model, nb_model = load_models()
    
    if vectorizer is None:
        st.stop()
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["Logistic Regression", "Naive Bayes"],
        help="Select which machine learning model to use for prediction"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Message")
        message = st.text_area(
            "Type your message here:",
            height=150,
            placeholder="Enter the message you want to classify...",
            help="Type any message to check if it's spam or ham"
        )
        
        # Predict button
        if st.button("🔍 Analyze Message", type="primary"):
            if message.strip():
                with st.spinner("Analyzing message..."):
                    result, confidence, probabilities = predict_with_confidence(
                        message, model_choice, vectorizer, lr_model, nb_model
                    )
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Result with color coding
                if result == "Spam":
                    st.error(f"🚨 **Prediction: {result}**")
                else:
                    st.success(f"✅ **Prediction: {result}**")
                
                # Confidence score
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence:.2f}%"
                )
                
                # Probability breakdown
                st.subheader("📈 Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Class': ['Ham', 'Spam'],
                    'Probability': [probabilities[0] * 100, probabilities[1] * 100]
                })
                
                st.bar_chart(prob_df.set_index('Class'))
                
                # Show probabilities as metrics
                col_ham, col_spam = st.columns(2)
                with col_ham:
                    st.metric("Ham Probability", f"{probabilities[0]*100:.2f}%")
                with col_spam:
                    st.metric("Spam Probability", f"{probabilities[1]*100:.2f}%")
                    
            else:
                st.warning("⚠️ Please enter a message to analyze.")
    
    with col2:
        st.subheader("ℹ️ About")
        st.info(
            f"""
            **Current Model:** {model_choice}
            
            **How it works:**
            1. Enter your message
            2. Choose a model
            3. Click 'Analyze Message'
            4. Get prediction with confidence
            
            **Models Available:**
            - Logistic Regression
            - Naive Bayes
            
            Both models are trained on SMS data with balanced classes.
            """
        )
        
        # Example messages
        st.subheader("📋 Try These Examples")
        
        example_spam = "Congratulations! You've won $1000! Click here to claim your prize now!"
        example_ham = "Hey, are we still meeting for lunch tomorrow at 12?"
        
        if st.button("Try Spam Example"):
            st.session_state.example_message = example_spam
            
        if st.button("Try Ham Example"):
            st.session_state.example_message = example_ham
            
        # Auto-fill example if button clicked
        if 'example_message' in st.session_state:
            st.text_area("Example:", value=st.session_state.example_message, height=100, disabled=True)

if __name__ == "__main__":
    main()
