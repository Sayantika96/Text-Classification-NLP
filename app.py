import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model pipeline and label encoder
pipeline = joblib.load("best_pipeline.pkl")  # Replace with your actual saved pipeline file
le = joblib.load("label_encoder.pkl")

# Function to preprocess and predict the text
def predict_category(text):
    # Use the pre-trained pipeline to predict the category
    prediction = pipeline.predict([text])  # Predict category for the input text
    # Decode the label back to the original category
    predicted_category = le.inverse_transform(prediction)[0]
    return predicted_category

# Streamlit App UI
st.title("Text Classification App")
st.write("""
This is a simple app for text classification. It predicts the category of a given text.
""")

# Input for text
input_text = st.text_area("Enter Text", "Type your text here...")

if st.button("Classify"):
    if input_text:
        # Predict the category
        category = predict_category(input_text)
        st.write(f"The predicted category is: **{category}**")
    else:
        st.write("Please enter a text for classification.")

# Optionally, display model metrics (you can skip this in a production version)
if st.button("Show Model Metrics"):
    # Load your dataset for training/testing (optional, for showing metrics)
    data = pd.read_csv("output_chunk_1.csv")  # Use your actual dataset

    # Preprocessing the dataset (if necessary)
    X = data["text"]  # Replace with the correct text column
    y = data["category"]  # Replace with the target column

    # Encode labels (for evaluating on the full dataset)
    y_encoded = le.transform(y)

    # Evaluate the model
    y_pred = pipeline.predict(X)
    accuracy = (y_encoded == y_pred).mean()
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.write("Classification Report:")
    from sklearn.metrics import classification_report
    st.text(classification_report(y_encoded, y_pred))
