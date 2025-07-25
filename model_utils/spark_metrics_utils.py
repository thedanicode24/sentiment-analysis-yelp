import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def display_metrics_table(data, title="Performance Metrics Table"):
    """
    Displays a formatted table of performance metrics for multiple classification models, 
    highlighting the best score in each metric column with a colored cell.

    Parameters
    ----------
    data : list of dict
        A list of dictionaries where each dictionary represents the metrics of a model.
        Each dictionary must contain a "Model" key and the following metric keys:
        "Accuracy", "Precision (0)", "Precision (1)", "Recall (0)", "Recall (1)", "Weighted F1".

    title : str, optional
        The title displayed above the table. Default is "Performance Metrics Table".

    Output
    ------
    A matplotlib plot containing:
        - A table with rows representing different models
        - Columns for each metric (accuracy, precision, recall, F1 score)
        - Rounded metric values (3 decimals)
        - Highlighted cells in light green for the best-performing model per metric
        - Clear formatting with borders and readable font size

    Notes
    -----
    - The table is purely visual and intended for comparing performance across models.
    - The best value in each column is highlighted using a custom green shade.
    - The function automatically adjusts table height based on the number of models.
    """
    
    metrics = ["Accuracy", "Precision (0)", "Precision (1)", "Recall (0)", "Recall (1)", "Weighted F1"]
    df = pd.DataFrame(data).set_index("Model")
    data = df[metrics]

    fig, ax = plt.subplots(figsize=(9, max(3, len(df)*0.4)))
    ax.set_axis_off()

    table = ax.table(cellText=data.round(3).values,
                     rowLabels=data.index,
                     colLabels=metrics,
                     cellLoc='center',
                     rowLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_facecolor('white')

    for col_idx, col in enumerate(metrics):
        col_values = data[col]
        max_val = col_values.max()
        min_val = col_values.min()
        max_row_label = col_values.idxmax()
        row_idx = list(data.index).index(max_row_label)

        normalized = (max_val - min_val) / (max_val - min_val + 1e-9)

        base_r = int(255 - 100 * normalized)
        base_g = 255
        base_b = int(255 - 100 * normalized)
        highlight_color = f'#{base_r:02x}{base_g:02x}{base_b:02x}'

        cell = table[(row_idx + 1, col_idx)]
        cell.set_facecolor(highlight_color)

    plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

def print_metrics(metrics):
    """
    Prints a tabular summary of binary classification metrics computed from a 
    PySpark MulticlassMetrics object.

    Parameters
    ----------
    metrics : pyspark.mllib.evaluation.MulticlassMetrics
        An object that provides evaluation metrics for a binary classification model, 
        such as accuracy, precision, recall, and F1-score for each class (0 and 1), 
        as well as the weighted F1 score.

    Output
    ------
    Prints a formatted table with the following metrics:
        - Accuracy (overall correct predictions)
        - Precision (0 and 1) (positive predictive value for each class)
        - Recall (0 and 1) (true positive rate for each class)
        - F1 (0 and 1) (harmonic mean of precision and recall for each class)
        - Weighted F1 (F1-score weighted by class support)
    """

    data = {
        "Metric": [
            "Accuracy",
            "Precision (0)",
            "Precision (1)",
            "Recall (0)",
            "Recall (1)",
            "F1 (0)",
            "F1 (1)",
            "Weighted F1"
        ],
        "Score": [
            metrics.accuracy,
            metrics.precision(0.0),
            metrics.precision(1.0),
            metrics.recall(0.0),
            metrics.recall(1.0),
            metrics.fMeasure(0.0),
            metrics.fMeasure(1.0),
            metrics.weightedFMeasure()
        ]
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))


def confusion_matrix(metrics):
    """
    Plots the confusion matrix for a binary classification model using a heatmap with 
    absolute values and percentages.

    Parameters
    ----------
    metrics : pyspark.mllib.evaluation.MulticlassMetrics
        An object containing evaluation metrics for a classification model, including 
        the confusion matrix.

    Output
    ------
    Displays a heatmap of the confusion matrix with:
        - Absolute counts of predictions (true positives, false positives, etc.)
        - Corresponding percentages for each cell relative to the actual class
        - Labeled axes for true and predicted classes (Negative and Positive)
    
    The heatmap includes visual enhancements like:
        - Color-coded cells (Blues colormap)
        - Bold annotations for clarity
        - Grid lines for better separation of cells
    """
    conf_matrix = metrics.confusionMatrix().toArray()
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    percentages = conf_matrix / row_sums * 100

    labels = np.asarray([
        [f"{int(conf_matrix[i, j])}\n({percentages[i, j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        conf_matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=2,
        linecolor='gray',
        square=True,
        annot_kws={"size": 14, "weight": "bold", "color": "black"}
    )

    plt.title("Confusion Matrix", fontsize=16, weight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(ticks=[0.5, 1.5], labels=["Negative", "Positive"], fontsize=12)
    plt.yticks(ticks=[0.5, 1.5], labels=["Negative", "Positive"], rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()