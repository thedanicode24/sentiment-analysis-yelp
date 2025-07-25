import matplotlib.pyplot as plt
import numpy as np

def plot_dict_highlight_max(data: dict, title="Comparison Table", col_label="Key", val_label="Value"):
    """
    Plots a table from a dictionary, highlighting the row with the maximum value.

    Parameters:
    -----------
    data : dict
        A dictionary where keys are the row labels and values are the numerical data to display.
    title : str, optional
        The title of the plot (default is "Comparison Table").
    col_label : str, optional
        The label for the first column (default is "Key").
    val_label : str, optional
        The label for the second column (default is "Value").

    Returns:
    --------
    None
        Displays a matplotlib table with the maximum value row highlighted in green.
    """

    keys = list(data.keys())
    values = list(data.values())
    max_idx = values.index(max(values))

    fig, ax = plt.subplots(figsize=(6, len(keys)*0.5 + 1))
    ax.axis('off')

    cell_text = [[k, f"{v:.4f}"] for k, v in zip(keys, values)]
    col_labels = [col_label, val_label]

    table = ax.table(cellText=cell_text,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')

    for i in range(len(keys)):
        color = '#d0f0c0' if i == max_idx else 'white'
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 1)].set_facecolor(color)

    for j in range(2):
        table[(0, j)].set_fontsize(14)
        table[(0, j)].set_text_props(weight='bold', color='black')
        table[(0, j)].set_facecolor('#f1f1f1')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.title(title, fontsize=16, weight='bold')
    plt.show()
