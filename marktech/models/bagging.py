from typing import Dict, Any, Union
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, mean_squared_error
import numpy as np

# Local import
from marktech.search.grid_search import ccc


class BaggingModel:
    """Bagging of multiple model for PHQ-8 score regression."""

    def __init__(self, path_model: Path, d_model_cls: Dict[str, Any], threshold: float = 0.4):
        self.models = self.load_models(path_model, d_model_cls)
        self.threshold = threshold

    @staticmethod
    def load_models(path_model: Path, d_model_cls: Dict[str, Any]) -> Dict[str, Any]:
        """ Load all classifier.

        """
        d_models = {}
        for mdl_key, mdl_cls in d_model_cls.items():
            d_models[mdl_key] = mdl_cls.load(path_model / mdl_key)

        return d_models

    def predict(self,
            X_vgg_full: np.ndarray, X_lld_full: np.ndarray, X_vgg_seq: np.ndarray,
            X_lld_seq: np.ndarray
    ) -> np.ndarray:
        """ Predict new inputs proba for each class.

        """
        bagged_y = np.zeros(X_vgg_full.shape[0], dtype=float)
        for mdl_key, mdl in self.models.items():
            if 'vgg_full' in mdl_key:
                y = mdl.predict(X_vgg_full)
            elif 'lld_full' in mdl_key:
                y = mdl.predict(X_lld_full)
            elif 'vgg_seq' in mdl_key:
                y = mdl.predict(X_vgg_seq)
            elif 'lld_seq' in mdl_key:
                y = mdl.predict(X_lld_seq)
            else:
                raise ValueError(f'{mdl_key} key is not valid')

            bagged_y += y

        return bagged_y / len(self.models)

    def evaluate(
            self, X_vgg_full: np.ndarray, X_lld_full: np.ndarray, X_vgg_seq: np.ndarray,
            X_lld_seq: np.ndarray, y: np.ndarray, include_graph: bool = False
    ) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
        """ Evaluate a model using most trendy performance metricx (ROC, Accuracy, ...)

        """
        y_phat = self.predict(X_vgg_full, X_lld_full, X_vgg_seq, X_lld_seq)
        y_hat = y_phat > self.threshold

        # Compute regression metric
        rmse = mean_squared_error(y_phat, y, squared=False)
        cccoef = ccc(y_phat, y)

        # Get train roc auc
        fpr, tpr, _ = roc_curve(y > 0.4, y_phat)
        roc_auc = auc(fpr, tpr)

        d_info = {
            'rmse': round(float(rmse), 3),
            'ccc': round(float(cccoef), 3),
            'accuracy': round(float(accuracy_score(y > 0.4, y_hat)), 3),
            'roc_auc': round(float(roc_auc), 3),
        }
        if include_graph:
            return {
                **d_info, 'cm': confusion_matrix(y > 0.4, y_hat),
                'roc_auc': {'fpr': fpr, 'tpr': tpr, 'roc_auc': round(float(roc_auc), 3)},
            }

        return d_info

