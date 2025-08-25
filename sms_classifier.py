# Importing the required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Loading the dataset
train_data = pd.read_csv('/content/train-data.tsv', sep='\t')
test_data = pd.read_csv('/content/test-data.tsv', sep='\t')

# Let's quickly check the first few rows
print("Sample training data:")
print(train_data.head())

# Building the model inside a pipeline:
# Step 1: Convert text into numerical features using TF-IDF
# Step 2: Apply a Naive Bayes classifier to those features
model = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')), 
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(train_data['message'], train_data['label'])

# Evaluate on the test set
predictions = model.predict(test_data['message'])
print("\nModel Accuracy on Test Data:", accuracy_score(test_data['label'], predictions))

# Define the prediction function
def predict_message(message):
    """
    Predict whether an SMS message is 'ham' or 'spam'.
    Returns: [probability, label]
    """
    # Get prediction probabilities
    probs = model.predict_proba([message])[0]
    
    # Index 0 = ham, index 1 = spam
    spam_prob = probs[1]
    
    # Decide label
    label = "spam" if spam_prob >= 0.5 else "ham"
    
    return [spam_prob, label]

# Quick test cases
print("\nTest Cases:")
print(predict_message("Congratulations! You won a free ticket to Bahamas. Call now!"))
print(predict_message("Hey, are we still on for lunch today?"))
