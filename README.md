# Sentiment Analysis on Yelp Reviews with PySpark

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-Colab-lightgrey)

This project implements a sentiment analysis pipeline on Yelp review data using Apache Spark and PySpark. The objective is to classify user reviews into positive or negative sentiments by leveraging natural language processing and machine learning techniques.

---

### Dataset and Setup
- The Yelp review dataset was downloaded from Kaggle.
- Initially, only **1% of the dataset** was used to speed up experimentation and model tuning.
- The entire dataset was then used to train the final model after model selection.

---

### Goal
- Classify Yelp reviews into **Positive (1)** or **Negative (0)** sentiment classes.

---

## 🧪 Workflow and Methodology

1. **Data Preprocessing:**
   - Custom text cleaning transformer to lowercase text and remove punctuation.
   - Tokenization of the cleaned text.
   - Removal of stopwords.
   - Stemming using Snowball Stemmer.
   - Conversion to TF-IDF feature vectors.

2. **Model Training and Evaluation:**
   - The 1% dataset was split into train/test subsets.
   - Three machine learning algorithms were trained and evaluated:
     - Logistic Regression
     - Naive Bayes
     - Linear Support Vector Classifier (Linear SVC)
   - Hyperparameter tuning was done using 3-fold cross-validation.
   - Models were evaluated mainly based on the **Weighted F1 Score** to handle class imbalance.

3. **Model Selection:**
   - The model with the highest Weighted F1 Score was chosen.

4. **Final Model Training:**
   - The selected model was retrained on the **entire Yelp dataset** to maximize learning.

5. **Testing on New Reviews:**
   - The final trained model and preprocessing pipeline were saved and later loaded.
   - Predictions were made on new, unseen review texts to verify performance.

---

## 📊 Evaluation Metrics and Visualization

- The main evaluation metric used is the **Weighted F1 Score**, balancing precision and recall across classes.
- A comparison table of the models’ metrics was created.
- Confusion matrices and other visualizations were produced to better understand model performance.

---

## 📁 Project Structure

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

---

## 🛠 Technologies and Libraries Used

- **Apache Spark** (PySpark) for scalable data processing and machine learning.
- **NLTK** for text preprocessing (stopwords, stemming).
- **Pandas** for tabular data handling.
- **Matplotlib** and **Seaborn** for plotting and visualization.
- **Google Colab** as the development and execution environment.

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
