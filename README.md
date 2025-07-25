# Sentiment Analysis on Yelp Reviews with PySpark

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-Colab-lightgrey)

This project implements a sentiment analysis pipeline on Yelp review data using Apache Spark and PySpark. The objective is to classify user reviews into positive or negative sentiments by leveraging natural language processing and machine learning techniques.

## 📌 Overview

- **Dataset**: Yelp Review Dataset (1% sample initially, full dataset used after model selection).
- **Framework**: PySpark, executed on Google Colab.
- **Goal**: Classify reviews as Positive (1) or Negative (0).
- **Models Used**:
  - Logistic Regression
  - Naive Bayes
  - Linear Support Vector Classifier (SVC)

## 🧪 Workflow

1. **Preprocessing**:
   - Custom transformer for text cleaning (lowercasing, punctuation removal)
   - Tokenization
   - Stopword removal
   - Stemming (Snowball Stemmer)
   - TF-IDF vectorization

2. **Model Training and Evaluation**:
   - Split dataset into train/test
   - Cross-validation (3 folds) to tune hyperparameters
   - Evaluation using **F1 score**

3. **Model Comparison**:
   - Compared F1 scores
   - Select the best model

4. **Final Training**:
   - The best model was retrained on the **entire dataset**

5. **Custom Testing**:
   - Loaded the saved preprocessing pipeline and trained model
   - Applied to new unseen review texts for prediction

## 📊 Evaluation

- All models were evaluated using the **Weighted F1 Score**, to account for class imbalance and give a balanced view of performance.
- A comparison table and a visualization were generated to identify the best-performing model.

## 📁 Project Structure

## Project Structure

- `sentiment-analysis-yelp.ipynb`  
  Main notebook where the pipeline is run and models are trained/tested

- `requirements.txt`  
  Python dependencies

- `README.md`  
  Project overview and documentation

- `data_balancing/`  
  Module for dataset balancing techniques  
  └── `data_balancing.py`

- `final_model/`  
  Code for training the final chosen model on full dataset  
  └── `final_model.py`

- `model_utils/`  
  Utility scripts for model evaluation and metrics  
  ├── `model_utils.py`  
  └── `spark_metrics_utils.py`

- `pipelines/`  
  Definitions of ML pipelines for different models  
  ├── `linear_svc.py`  
  ├── `naive_bayes.py`  
  └── `logistic_regression.py`

- `preprocessing/`  
  Text preprocessing scripts and custom transformers  
  ├── `preprocessing.py`  
  └── `text_preprocessing.py`

- `utils/`  
  General utility functions used throughout the project  
  ├── `colab_utils.py`  
  ├── `io_utils.py`  
  ├── `table_utils.py`  
  └── `time_utils.py`

- `models/`  
  Saved trained models and preprocessing pipelines  
  ├── `logistic_regression_model/`  
  ├── `naive_bayes_model/`  
  ├── `linear_svc_model/`  
  ├── `final_model/`  
  └── `preprocessing_model/`

---

## 🔧 Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 🧰 Key Libraries

- **PySpark** – Distributed computing & ML pipelines  
- **NLTK** – Natural language preprocessing (stemming, stopwords, etc.)  
- **Pandas** – Tabular data manipulation and summary  
- **Matplotlib** – Visualizations and metric plotting  
- **Seaborn** – Enhanced plots and heatmaps

---

## 💡 Future Improvements

- Implement advanced deep learning models (e.g. BERT, LSTM via Spark NLP or external integrations)  
- Perform hyperparameter tuning with more folds and expanded parameter ranges  
- Deploy the best model through a RESTful API or web app for real-time sentiment prediction  
- Add model explainability tools (e.g. SHAP or LIME for feature interpretation)

---

## 📜 License

This project is open-source and licensed under the **MIT License**.  
Feel free to use, modify, and distribute it responsibly.

---

## 🚀 Author

**Daniel**  
Computer Engineering Student  

Contributions, stars, and forks are always welcome! ⭐  
Feel free to reach out or suggest improvements.

---

## 📦 Requirements

To install all dependencies, run:

```bash
pip install -r requirements.txt
```
