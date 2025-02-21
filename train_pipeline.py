import joblib
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Load Dataset
df = pd.read_csv("output_chunk_1.csv")  # Replace with actual dataset path

# Basic Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text  # Directly return preprocessed text

df['cleaned_text'] = df["text"].apply(preprocess_text)

# Feature & Target Split
X = df['cleaned_text']
y = df['category']  # Replace with actual category column

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "MLP Neural Network": MLPClassifier()
}

# Hyperparameter Grids
param_grids = {
    "Logistic Regression": {'model__C': np.logspace(-3, 2, 6), 'model__solver': ['liblinear', 'lbfgs']},
    "SVM": {'model__C': [0.1, 1, 10, 100], 'model__kernel': ['linear', 'rbf'], 'model__gamma': ['scale', 'auto']},
    "Random Forest": {'model__n_estimators': [50, 100, 200], 'model__max_depth': [10, 20, 30, None]},
    "XGBoost": {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.01, 0.1, 0.2]},
    "MLP Neural Network": {'model__hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)], 'model__activation': ['relu', 'tanh']}
}

# Store the Best Model
best_model = None
best_score = 0

# Optimize Using Pipeline with TF-IDF
for model_name, model in models.items():
    print(f"Tuning {model_name}...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('model', model)
    ])

    param_grid = param_grids.get(model_name, None)
    
    if param_grid:
        search = RandomizedSearchCV(pipeline, param_grid, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        model_score = search.best_score_
        tuned_model = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        model_score = pipeline.score(X_test, y_test)
        tuned_model = pipeline

    print(f"{model_name} - Best Accuracy: {model_score:.4f}")

    # Keep the best performing model
    if model_score > best_score:
        best_model = tuned_model
        best_score = model_score

# Save the Best Model
joblib.dump(best_model, "best_pipeline.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Best model and label encoder saved successfully.")
