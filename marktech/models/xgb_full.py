# Global import
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
import pickle
import json
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt
import numpy as np
import xgboost

# local import / global var


class XGBRegressor:
    """Extreme Gradient boosting Regressor"""

    def __init__(
            self, feature_names: List[str], categorical_features: List[str], is_fitted: bool = False,
            xgb_params: Dict[str, Union[str, int, float]] = None, encoders: Dict[str, Any] = None,
            clf: xgboost.XGBRegressor = None
    ):
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.encoders = {}
        xgb_params = xgb_params or {}

        if encoders is not None:
            self.encoders = encoders

        if clf is not None:
            self.clf = clf
        else:
            self.clf = xgboost.XGBRegressor(use_label_encoder=False, **xgb_params)

        self.is_fitted = is_fitted

    @staticmethod
    def load(path_dir: Path) -> 'XGBRegressor':

        with open(path_dir / 'meta.json', 'r') as handle:
            d_meta = json.load(handle)

        with open(path_dir / 'encoders.pckl', 'rb') as handle:
            encoders = pickle.load(handle)

        with open(path_dir / 'xgb.pckl', 'rb') as handle:
            clf = pickle.load(handle)

        return XGBRegressor(
            d_meta['feature_names'], d_meta['categorical_features'], d_meta['is_fitted'], encoders=encoders,
            clf=clf
        )

    def save(self, path_dir: Path) -> None:
        d_meta = {
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'is_fitted': self.is_fitted
        }
        path_dir.mkdir()
        with open(path_dir / 'meta.json', 'w') as handle:
            json.dump(d_meta, handle)

        with open(path_dir / 'encoders.pckl', 'wb') as handle:
            pickle.dump(self.encoders, handle)

        with open(path_dir / 'xgb.pckl', 'wb') as handle:
            pickle.dump(self.clf, handle)

    def fit_encoders(self, X: np.ndarray) -> None:
        for col in self.categorical_features:
            i = self.feature_names.index(col)
            self.encoders[col] = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            ).fit(X[:, [i]])

    def encode(self, X: np.ndarray) -> np.ndarray:
        if not self.encoders:
            return X

        # Encode X columns without loosing column order
        for col in self.categorical_features:
            i = self.feature_names.index(col)
            X[:, i] = self.encoders[col].transform(X[:, [i]])[:, 0]

        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBRegressor':
        # Fit encoders and transform input
        self.fit_encoders(X)

        X_encoded = self.encode(X.copy())
        self.clf.fit(X_encoded, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('Classifier not fitted yet.')

        X_encoded = self.encode(X.copy())
        y_hat = self.clf.predict(X_encoded, validate_features=False)

        return y_hat

    def plot_feature_importance(self, max_num_features: int = None, figsize: Tuple[int, int] = None):

        # Create figure
        plt.figure(figsize=figsize)
        ax = plt.subplot()

        self.clf.get_booster().feature_names = self.feature_names
        xgboost.plot_importance(self.clf.get_booster(), ax=ax, max_num_features=max_num_features)
        plt.show()

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)




