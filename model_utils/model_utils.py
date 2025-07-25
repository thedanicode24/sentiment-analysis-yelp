"""
model_utils.py

Utility functions for building pipelines, parameter grids, and cross-validation in PySpark.
"""

import os
from pyspark.ml import Model, Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import Evaluator

def build_pipeline(stages):
    """
    Builds a Spark ML pipeline from a list of stages.

    Args:
        stages (List[PipelineStage]): List of preprocessing and model stages.

    Returns:
        Pipeline: Spark ML pipeline object.
    """
    return Pipeline(stages=stages)

def build_param_grid(model, param_dict):
    """
    Builds a parameter grid for hyperparameter tuning.

    Args:
        model: A PySpark estimator (e.g., LogisticRegression).
        param_dict (dict): Dictionary of parameter names (str) and lists of values.

    Returns:
        list: A list of ParamMap for grid search.
    """
    grid = ParamGridBuilder()
    for param, values in param_dict.items():
        grid = grid.addGrid(getattr(model, param), values)
    return grid.build()

def build_cross_validator(pipeline, param_grid, evaluator: Evaluator, folds=3):
    """
    Constructs a CrossValidator for model selection.

    Args:
        pipeline (Pipeline): The ML pipeline to evaluate.
        param_grid (list): List of ParamMap for hyperparameter tuning.
        evaluator (Evaluator): A PySpark evaluator (e.g., MulticlassClassificationEvaluator).
        folds (int): Number of cross-validation folds.

    Returns:
        CrossValidator: Configured CrossValidator object.
    """
    return CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=folds,
        parallelism=2
    )

def save_model(model: Model, path: str, model_name: str) -> None:
    """
    Save a PySpark ML model to a specified directory, creating the directory if it does not exist.

    Parameters:
        model (Model): The trained PySpark ML model to save.
        path (str): The base directory where the model will be saved.
        model_name (str): The name of the model directory to create within the base path.

    Returns:
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    model_path = os.path.join(path, model_name)
    
    model.save(model_path)
