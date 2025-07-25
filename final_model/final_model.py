"""
final_model.py

Trains the final selected model on the full dataset using best hyperparameters,
and evaluates it without cross-validation.
"""

import time
from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col, when

from data_balancing.data_balancing import add_class_weight
from preprocessing.preprocessing import build_preprocessing_pipeline
from utils.time_utils import print_from_seconds_to_hours
from config import evaluator

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel, NaiveBayesModel, LinearSVCModel
from pyspark.sql import SparkSession, DataFrame
from typing import List

def train_final_model(train_df, test_df, best_params: dict, model_name: str):
    """
    Trains and evaluates the final model (LR, NB, or SVC) using best hyperparameters.

    Args:
        train_df (DataFrame): Full training data.
        test_df (DataFrame): Test data.
        best_params (dict): Best hyperparameters from previous tuning.
        model_name (str): One of 'lr', 'nb', 'svc'.

    Returns:
        Tuple[Model, MulticlassMetrics, PipelineModel]: 
            - Trained model instance (LogisticRegressionModel, NaiveBayesModel, or LinearSVCModel).
            - MulticlassMetrics containing evaluation results.
            - Preprocessing PipelineModel used to transform data.
    """

    start = time.time()

    use_idf = model_name in ("lr", "svc")
    preprocessing_pipeline = build_preprocessing_pipeline(use_idf=use_idf)
    preprocessing_model = preprocessing_pipeline.fit(train_df)

    end = time.time()
    print("Total preprocessing: ", print_from_seconds_to_hours(end-start))

    train_df = preprocessing_model.transform(train_df)
    test_df = preprocessing_model.transform(test_df)

    start = time.time()
    
    if model_name in ("lr", "svc"):
        train_df = add_class_weight(train_df)

    if model_name=="lr":
        model = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            regParam=best_params["regParam"],
            elasticNetParam=best_params["elasticNetParam"]
        )

    elif model_name=="nb":
        model = NaiveBayes(
            featuresCol="features",
            labelCol="label",
            modelType="multinomial",
            smoothing=best_params["smoothing"]
        )

    elif model_name=="svc":
        model = LinearSVC(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            regParam=best_params["regParam"],
            maxIter=best_params["maxIter"]
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    trained_model = model.fit(train_df)

    end = time.time()
    print("Total training: ", print_from_seconds_to_hours(end-start))

    predictions = trained_model.transform(test_df)
    pred_and_labels = predictions.select("prediction", "label") \
        .rdd.map(lambda row: (float(row['prediction']), float(row['label'])))
    metrics = MulticlassMetrics(pred_and_labels)

    return trained_model, metrics, preprocessing_model

def load_model(model_type: str, model_path: str):
    """
    Load a Spark ML classification model based on the specified type.

    Args:
        model_type (str): The type of model to load ("lr", "nb", "svc").
        model_path (str): Path to the saved model.

    Returns:
        Model: An instance of the loaded Spark ML model.

    Raises:
        ValueError: If the model type is unsupported.
    """
    if model_type == "lr":
        return LogisticRegressionModel.load(model_path)
    elif model_type == "nb":
        return NaiveBayesModel.load(model_path)
    elif model_type == "svc":
        return LinearSVCModel.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")

def preprocess_text(text_list: List[str], spark_session: SparkSession, preprocessing_model_path: str) -> DataFrame:
    """
    Preprocess a list of texts using a saved Spark ML preprocessing pipeline.

    Args:
        text_list (List[str]): List of input texts to preprocess.
        spark_session (SparkSession): Active Spark session.
        preprocessing_model_path (str): Path to the saved preprocessing model.

    Returns:
        DataFrame: A DataFrame with the transformed (preprocessed) data.
    """
    text_df = spark_session.createDataFrame([(r,) for r in text_list], ["text"])
    preprocessing_model = PipelineModel.load(preprocessing_model_path)
    return preprocessing_model.transform(text_df)

def show_predictions(predictions_df: DataFrame, model_type: str):
    """
    Display model predictions in a format specific to the model type.

    Args:
        predictions_df (DataFrame): DataFrame containing prediction results.
        model_type (str): Type of the model used ("lr", "nb", "svc").

    Raises:
        ValueError: If the model type is unsupported.
    """
    if model_type in ['lr', 'nb']:
        predictions_df.select("text", "probability", "prediction").show()
    elif model_type == 'svc':
        predictions_df.select("text", "rawPrediction", "prediction").show()
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")

def run_prediction(text_list: List[str], selected_model: str, preprocessing_model_path: str, final_model_path: str, spark_session: SparkSession):
    """
    Run the full prediction pipeline: preprocessing, model loading, inference, and result display.

    Args:
        text_list (List[str]): List of input texts to classify.
        selected_model (str): Type of model to use ("lr", "nb", "svc").
        preprocessing_model_path (str): Path to the saved preprocessing model.
        final_model_path (str): Path to the saved classification model.
        spark_session (SparkSession): Active Spark session.
    """
    preprocessed_df = preprocess_text(text_list, spark_session, preprocessing_model_path)
    model = load_model(selected_model, final_model_path)
    predictions = model.transform(preprocessed_df)
    show_predictions(predictions, selected_model)