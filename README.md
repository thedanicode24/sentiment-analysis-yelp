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
   - Evaluation using **F1 score** and other metrics (accuracy, precision, recall)

3. **Model Comparison**:
   - Compared F1 scores:
     - **Logistic Regression**: 0.9349
     - **Naive Bayes**: 0.8938
     - **Linear SVC**: 0.9372
   - Best model: **Linear SVC**

4. **Final Training**:
   - The best model was retrained on the **entire dataset**

5. **Custom Testing**:
   - Loaded the saved preprocessing pipeline and trained model
   - Applied to new unseen review texts for prediction

## 📊 Evaluation

- All models were evaluated using the **Weighted F1 Score**, to account for class imbalance and give a balanced view of performance.
- A comparison table and a visualization were generated to identify the best-performing model.

## 📁 Project Structure

sentiment-analysis-yelp/
│
├── sentiment-analysis-yelp.ipynb # Main notebook
├── requirements.txt # Dependencies
├── README.md # Project overview
├── models/ # Saved ML models
├── preprocessing/ # Custom preprocessing pipeline
│ ├── text_cleaner.py
│ └── stemmer.py
├── utils/ # Utility functions
│ └── metrics_plot.py
└── data/
└── yelp_subset.csv # Sample dataset


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
