# Text Classification Project - ADSC4720: Data Mining

This project is focused on text classification tasks, using multiple machine learning models such as Logistic Regression, Support Vector Machines (SVM), Random Forest, XGBoost, and Multi-Layer Perceptron (MLP) Neural Networks. Hyperparameter tuning is performed to optimize the models, and a React-based front-end is provided for real-time text classification.

## Project Structure

```
Text-Classification-NLP/
│
├── logs/                        # Logs from MLflow experiments
├── mlruns/                      # MLflow run data
├── 0.26.0/                      # MLflow specific version data
├── app.py                       # Flask API for model inference
├── best_pipeline.pkl            # Best performing pipeline model
├── label_encoder.pkl            # Label encoder used for encoding target variables
├── output_chunk_1.csv           # Data output after preprocessing
├── text_classification.ipynb    # Jupyter Notebook for exploration and experimentation
├── train_pipeline.py            # Python script to train and evaluate models
├── vectorizer.pkl               # Text vectorizer used in the model pipeline
└── xgboost_best_model.pkl       # Best trained XGBoost model
└── sample2percent.csv       # 2% data kept aside
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Text-Classification-NLP.git
cd Text-Classification-NLP
```

### 2. Install dependencies

To install Python dependencies for the backend, run:

```bash
pip install -r requirements.txt
```

For the React frontend, run the following inside the `ui` directory:

```bash
cd ui
npm install
```

### 3. Set up the backend

Ensure that your Python environment has the following libraries installed:

- pandas
- nltk
- scikit-learn
- xgboost
- joblib
- flask
- mlflow

### 4. Running the model training and evaluation script

Run the following Python script to train and evaluate the models:

```bash
python train_pipeline.py
```

This script will:
- Preprocess the text data.
- Train multiple models (Logistic Regression, SVM, Random Forest, XGBoost, MLP Neural Network).
- Perform hyperparameter tuning using RandomizedSearchCV.
- Save the best-performing model and pipeline to `.pkl` files.

### 5. Running the React front-end

To start the React front-end, run:

```bash
npm start
```

This will start a development server for the UI. You can now interact with the model using the web interface.

## Text Classification API

The API endpoint `app.py` is used to classify text based on the selected model.The response will contain the predicted category and accuracy data for each model.

## Models

The following models are trained and evaluated:

- **Logistic Regression**: A linear model for classification.
- **Support Vector Machine (SVM)**: A model that finds the hyperplane maximizing margin between classes.
- **Random Forest**: A collection of decision trees used for classification.
- **XGBoost**: A gradient boosting model for classification.
- **MLP Neural Network**: A multi-layer perceptron for deep learning classification.

### Hyperparameter Tuning

Hyperparameter tuning is performed using `RandomizedSearchCV` to find the best set of hyperparameters for each model.

## Model Evaluation

After training and tuning, the models' performance is evaluated on the test set, and accuracy scores are displayed. The best-performing model is saved as a `.pkl` file for future use.

## Requirements

- Python 3.7+
- Node.js and npm (for the React UI)
- Libraries listed in `requirements.txt` for the backend

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Scikit-learn for machine learning algorithms and tools.
- XGBoost for gradient boosting.
- React for building the front-end interface.
- NLTK for text preprocessing and tokenization.
