"""
logistic_regression.py

Defines and runs the logistic regression pipeline with preprocessing, hyperparameter tuning, 
and evaluation.
"""

import time
from pyspark.ml.classification import LogisticRegression
from config import PARAMS, evaluator
from preprocessing.preprocessing import get_preprocessing_stages
from data_balancing.data_balancing import add_class_weight
from model_utils.model_utils import build_pipeline, build_param_grid, build_cross_validator
from utils.time_utils import print_from_seconds_to_hours

def run_lr(train_df, test_df):
    """
    Trains and evaluates a Logistic Regression model using cross-validation.

    Args:
        train_df (DataFrame): Training Spark DataFrame.
        test_df (DataFrame): Test Spark DataFrame.
        f1_scores (dict): Dictionary to store F1 score results.

    Returns:
        dict: Contains the trained model, f1 score and best parameters.
    """
    start = time.time()
    
    train_df = add_class_weight(train_df)
    
    preprocessing = get_preprocessing_stages(use_idf=True)

    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="label", 
        weightCol="class_weight"
    )

    pipeline = build_pipeline(preprocessing + [lr])
    param_grid = build_param_grid(lr, PARAMS["lr"])
    cv = build_cross_validator(pipeline, param_grid, evaluator)

    model = cv.fit(train_df)
    predictions = model.transform(test_df)
    f1 = evaluator.evaluate(predictions)

    best_model = model.bestModel.stages[-1]

    end = time.time()
    print("Total time: ", print_from_seconds_to_hours(end-start))

    return {
        "model": best_model,
        "f1_score": f1,
        "best_params": {
            "regParam": best_model.getRegParam(),
            "elasticNetParam": best_model.getElasticNetParam()
        }
    }
