# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load the dataset (Assume you have a CSV file with 'text' and 'label' columns)
# The 'text' column contains the email content, and the 'label' column contains 'spam' or 'ham' (non-spam).
df = pd.read_csv('spam.csv')

# Step 3: Preprocessing
# Tokenize, remove stop words, special characters, and normalize the text

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# Function to preprocess and clean the text
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing to the 'text' column
# Apply preprocessing to the 'message' column (assuming 'message' is the correct column name)
df['cleaned_text'] = df['Message'].apply(preprocess_text)

# Step 4: Feature Extraction (Using CountVectorizer and TfidfTransformer)
# Convert the cleaned text into a bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])

# You can also apply TF-IDF transformation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

# Step 5: Prepare the Labels
y = np.where(df['Category'] == 'spam', 1, 0)  # 1 for spam, 0 for ham (non-spam)

# Step 6: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Step 7: Train the Gaussian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(X_train.toarray(), y_train)  # Convert sparse matrix to dense array for Naive Bayes

# Step 8: Make Predictions and Evaluate the Model
y_pred = gnb.predict(X_test.toarray())

# Step 9: Evaluate the Model Performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Step 10: Classify a New Email (Function to classify new email)
def classify_new_email(new_email):
    # Preprocess the new email text
    cleaned_email = preprocess_text(new_email)

    # Transform the cleaned email using the vectorizer and TF-IDF transformer
    email_vector = vectorizer.transform([cleaned_email])
    email_tfidf = tfidf_transformer.transform(email_vector)

    # Predict using the trained model
    prediction = gnb.predict(email_tfidf.toarray())

    # Return the result as "spam" or "ham"
    if prediction == 1:
        return "Spam"
    else:
        return "Ham"


# Example usage: Classify a new email
new_email = "are you sure about this "
result = classify_new_email(new_email)
print(f"The email is classified as: {result}")


