"""
config.py

Stores parameter grids and shared evaluator for PySpark ML models.
"""

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

PARAMS = {
    "lr": {
        "regParam": [0.01, 0.1, 1.0],
        "elasticNetParam": [0.0, 0.5, 1.0]
    },
    "nb": {
        "smoothing": [0.5, 1.0, 1.5],
    },
    "svc": {
        "regParam": [0.01, 0.1, 1.0],
        "maxIter": [50, 100]
    }
}

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
