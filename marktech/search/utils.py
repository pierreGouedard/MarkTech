# Global import
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Union, Any
from tensorflow.python.data import Dataset
from pathlib import Path
import numpy as np
import threading
import shutil
import yaml

# local import
from marktech.models.datasets import DatasetMeta


def dataset2numpy(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """ Transform a tensorflow dataset into a n-dim numpy array.

    """
    l_X, l_y = [], []
    for X, y in dataset.as_numpy_iterator():
        l_X.append(X)
        l_y. append(y)

    return np.stack(l_X), np.array(l_y)


def ccc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    vx, cov_xy, cov_xy, vy = np.cov(y_pred, y_true, bias=True).flat
    mx, my = y_pred.mean(), y_true.mean()
    return 2*cov_xy / (vx + vy + (mx - my)**2)


def evaluate(
        mdl: Any, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray, weights: Tuple[float, float], display: bool = True
) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
    """ Evaluate a model using most trendy performance metricx (ROC, Accuracy, ...)

    """
    y_hat_train, y_hat_test = mdl.predict(X_train), mdl.predict(X_test)

    # Mean square error
    rmse_train = mean_squared_error(y_hat_train, y_train, squared=False)
    rmse_test = mean_squared_error(y_hat_test, y_test, squared=False)

    # Mean absolute error
    mae_train = mean_absolute_error(y_hat_train, y_train)
    mae_test = mean_absolute_error(y_hat_test, y_test)

    # R2 score
    r2_train = r2_score(y_train, y_hat_train)
    r2_test = r2_score(y_test, y_hat_test)

    # Weighted mean square error
    ax_weights = ((y_train - y_hat_train) < 0) * weights[0] + ((y_train - y_hat_train) >= 0) * weights[1]
    wmse_train = mean_squared_error(y_hat_train, y_train, sample_weight=ax_weights, squared=False)
    ax_weights = ((y_test - y_hat_test) < 0) * weights[0] + ((y_test - y_hat_test) >= 0) * weights[1]
    wmse_test = mean_squared_error(y_hat_test, y_test, sample_weight=ax_weights, squared=False)

    # Correlation
    ccc_train = ccc(y_hat_train, y_train)
    ccc_test = ccc(y_hat_test, y_test)

    # Gather metrics
    d_info = {
        'train_rmse': round(float(rmse_train), 3),
        'test_rmse': round(float(rmse_test), 3),
        'train_mae': round(float(mae_train), 3),
        'test_mae': round(float(mae_test), 3),
        'r2_train': round(float(r2_train), 3),
        'r2_test': round(float(r2_test), 3),
        'train_wmse': round(float(wmse_train), 3),
        'test_wmse': round(float(wmse_test), 3),
        'train_ccc': round(float(ccc_train), 3),
        'test_ccc': round(float(ccc_test), 3),
    }
    if display:
        print(d_info)

    return d_info


def init_search(
        key_metric: str, search_path: Path, model_path: Path = None, cls_model: Any = None, worst_metric: float = 0.
) -> Tuple[float, Any, Dict[str, Any]]:
    if search_path.exists():
        with open(search_path, 'r') as handle:
            d_info = yaml.safe_load(handle)

        if model_path is None:
            best_metric, best_model = worst_metric, None
        elif model_path.exists():
            best_metric = max([v[key_metric] for v in d_info.values()])
            best_model = cls_model.load(model_path)
        else:
            best_metric, best_model = worst_metric, None
    else:
        best_metric, best_model, d_info = worst_metric, None, {}

    return best_metric, best_model, d_info


def backup_search(
        search_path: Path, d_info: Dict[str, Any],  model_path: Path = None, best_model: Any = None,
        save_model: bool = False
) -> None:
    if model_path is not None:
        if model_path.exists() and save_model:
            shutil.rmtree(model_path.as_posix())
            best_model.save(model_path)

    with open(search_path, 'w') as handle:
        yaml.dump(d_info, handle, default_flow_style=False)


class FitThread(threading.Thread):

    def __init__(
            self, best_metric: float, target_metric: str, model_type: str, params: Any, cls_model: Any,
            weights: Tuple[float, float], X_train: np.ndarray = None, y_train: np.ndarray = None,
            meta_ds: DatasetMeta = None, train_ds: Dataset = None, X_test: np.ndarray = None, y_test: np.ndarray = None
    ):
        threading.Thread.__init__(self)

        self.model_type = model_type
        self.params = params
        self.cls_model = cls_model
        self.best_metric = best_metric
        self.target_metric = target_metric
        self.weights = weights

        # Training inputs
        self.X_train, self.y_train = X_train, y_train
        self.meta_ds, self.train_ds = meta_ds, train_ds

        # Test inputs
        self.X_test, self.y_test = X_test, y_test

        #
        self.mdl, self.metrics = None, None

    def run(self):
        if self.model_type == 'xgb':
            l_feature_names = [f'activation_{i}' for i in range(self.X_train.shape[1])]
            mdl = self.cls_model(l_feature_names, [], **{'xgb_params': self.params}) \
                .fit(self.X_train, self.y_train)
            self.metrics = evaluate(mdl, self.X_train, self.y_train, self.X_test, self.y_test, self.weights)
        else:
            mdl = self.cls_model(self.meta_ds, **self.params) \
                .fit(self.train_ds)
            self.metrics = evaluate(mdl, self.X_train, self.y_train, self.X_test, self.y_test, self.weights)

        test_metric = self.metrics[self.target_metric]
        train_metric = self.metrics[self.target_metric.replace('test', 'train')]

        if self.best_metric < test_metric and train_metric >= (test_metric / 3):
            self.mdl = mdl
            self.best_metric = test_metric

