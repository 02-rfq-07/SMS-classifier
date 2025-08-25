📱SMS SPAM CLASSIFIER

This project is a machine learning classifier that predicts whether an SMS message is ham (normal) or spam (unwanted advertisement or promotional message).

We use the SMS Spam Collection dataset, which has already been divided into training and testing sets. The goal is to build a model that can accurately classify unseen messages.

🚀 Project Overview

Input: A text message (string).

Output:

A list with two elements:

A probability score between 0 (ham) and 1 (spam).

A label: "ham" or "spam".

Example:

predict_message("Congratulations! You’ve won a free ticket 🎉")

Output → [0.92, 'spam']

🛠️ Steps in the Project

Data Loading:

Import the pre-split training and testing dataset.

Preprocessing:

Clean the text (lowercasing, removing special characters, etc.).

Convert words into numerical features using TF-IDF Vectorization.

Model Training:

Train a Naive Bayes classifier (works great for text classification).

Evaluation:

Measure performance on the test set (accuracy, precision, recall, F1-score).

Prediction Function:

A predict_message(message) function that returns probability and label.

📂 Project Structure
📦 SMS-Spam-Classifier
 ┣ 📜 main.py   # main file
 ┣ 📜 README.md                   # Project documentation
 

📊 Model Performance

Accuracy: ~97%

The model handles short SMS-style text effectively.
