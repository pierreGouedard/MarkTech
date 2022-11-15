# Global import
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from numpy import ndarray, arange
import pandas as pd
import numpy as np

# local import / global var
default_figsize = (9, 7)
l_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def lines_plot(
        x: np.ndarray, l_y: List[np.ndarray], l_labels: List[str], title: str, x_label: str, y_label: str
) -> None:
    """
    """
    for i, (y, label) in enumerate(zip(*[l_y, l_labels])):
        plt.plot(x, y, l_colors[i], label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def bar_plot(
        x: ndarray, y: ndarray, title: str, xlabel: str, ylabel: str, rotation: Optional[int] = 0,
        figsize: Optional[Tuple[int, int]] = None, limit_axis: Optional[Tuple[bool, bool]] = (False, False)
) -> None:
    """
    """
    plt.figure(figsize=figsize or default_figsize)
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=rotation)

    if limit_axis[0]:
        plt.xlim(x.min() - (abs(x.min()) * 0.001), x.max() + (abs(x.max()) * 0.001))

    if limit_axis[1]:
        plt.ylim(y.min() - (abs(y.min()) * 0.001), y.max() + (abs(y.max()) * 0.001))

    plt.show()


def pandas_hist(
        df: pd.DataFrame, col_name: str, title: str, xlabel: str, ylabel: str, is_kde: bool,
        figsize: Optional[Tuple[int, int]] = None, is_num: bool = False
) -> None:
    """
    """
    plt.figure(figsize=figsize or default_figsize)

    if not is_num:
        df[col_name].plot.hist()
    else:
        df[col_name].value_counts().plot(kind='bar')

    if is_kde:
        df[col_name].plot.kde(secondary_y=True)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def pandas_kde(
        df: pd.DataFrame, col_name: str, title: str, xlabel: str, ylabel: str, bw_method: float = 1.,
        figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    """
    # Build figure
    plt.figure(figsize=figsize or default_figsize)
    df[col_name].plot.kde(bw_method=bw_method)

    # Add information
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def plot_confusion_matrix(
        confusion_matrix: ndarray, class_names: List[str], title: str, show_value: bool = False,
        figsize: Optional[Tuple[int, int]] = None
) -> None:
    # Plot
    plt.figure(figsize=figsize or default_figsize)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Wistia)

    # Set class names
    tick_marks = arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add information
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add text with values if necessary
    if show_value:
        s = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(confusion_matrix[i][j]))

    plt.show()


def plot_roc_auc(
        fpr: np.array, tpr: np.array, roc_auc: float, title: str, figsize: Optional[Tuple[int, int]] = None
):
    plt.figure(figsize=figsize or default_figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
