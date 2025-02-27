import joblib
import re
import pandas as pd

# Load the trained model and label encoder
best_model = joblib.load("best_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load new data
df_new = pd.read_csv("sample2percent.csv") 
# Apply text preprocessing
df_new['cleaned_text'] = df_new["text"].apply(preprocess_text)

# Make predictions
predictions = best_model.predict(df_new['cleaned_text'])

# Convert numerical predictions back to original labels
predicted_labels = label_encoder.inverse_transform(predictions)

# Save predictions to CSV
df_new['predicted_category'] = predicted_labels
df_new.to_csv("predictions.csv", index=False)

print("Predictions saved successfully.")
