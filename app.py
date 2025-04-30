import streamlit as st
import joblib
import os

# Load model and vectorizer
model_path = 'models/logistic_model.pkl'
vectorizer_path = 'models/vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model or vectorizer not found. Please run main.py to train and save them.")
else:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Streamlit app
    st.title("📰 Fake News Detection")

    user_input = st.text_area("Enter the news article text:")

    if st.button("Predict"):
        if user_input:
            # Vectorize input
            input_tfidf = vectorizer.transform([user_input])

            # Predict
            prediction = model.predict(input_tfidf)[0]

            # Display result
            if prediction == 0:
                st.success("✅ The news is Real.")
            else:
                st.error("🚫 The news is Fake.")
        else:
            st.warning("Please enter some text to predict.")
