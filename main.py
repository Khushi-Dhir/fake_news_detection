import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
fake_df = pd.read_csv('data/fake.csv')
true_df = pd.read_csv('data/true.csv')

# Add a 'label' column to differentiate the two datasets
fake_df['label'] = 'fake'
true_df['label'] = 'true'

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Replace 'fake' and 'true' with 0 and 1 for the labels
df['label'] = df['label'].map({'fake': 0, 'true': 1})

# Features and Labels
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Print the accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model and vectorizer
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model and vectorizer saved successfully.")
