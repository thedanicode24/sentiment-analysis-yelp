"""
data_balancing.py

Contains utility functions for handling class imbalance in classification tasks.
"""

from pyspark.sql.functions import col, when

def add_class_weight(df, label_col="label", weight_col="class_weight"):
    """
    Adds a class weight column to the DataFrame to handle class imbalance.

    Args:
        df (DataFrame): Input Spark DataFrame with a label column.
        label_col (str): Name of the label column (default is 'label').
        weight_col (str): Name of the new weight column to add.

    Returns:
        DataFrame: New DataFrame with the weight column added.
    """
    n_samples = df.count()
    n_neg = df.filter(col(label_col) == 0).count()
    neg_ratio = round(n_neg / n_samples, 2)

    df = df.withColumn(
        weight_col,
        when(col(label_col) == 0, 1 - neg_ratio).otherwise(neg_ratio)
    )
    return df
