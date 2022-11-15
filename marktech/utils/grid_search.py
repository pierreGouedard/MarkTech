# Global import
from typing import List, Dict, Tuple, Union, Callable, Any
from pathlib import Path
import itertools
import threading
import shutil
import yaml
import gc
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error, confusion_matrix

# local import
from marktech.models.datasets import DatasetMeta
from marktech.models.xgb_full import XGBRegressor
from marktech.models.tf_full import TfModelFull
from marktech.models.tf_seq import TfModelSeq


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
        mdl: Union[TfModelFull, TfModelSeq, XGBRegressor], X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray, threshold: float = 10. / 25., include_graph: bool = False,
        display: bool = True
) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
    """ Evaluate a model using most trendy performance metricx (ROC, Accuracy, ...)

    """
    y_phat_train, y_phat_test = mdl.predict(X_train), mdl.predict(X_test)
    y_hat_train, y_hat_test = (y_phat_train > threshold).astype(int), (y_phat_test > threshold).astype(int)

    # Compute regression metric
    rmse_train = mean_squared_error(y_phat_train, y_train, squared=False)
    rmse_test = mean_squared_error(y_phat_test, y_test, squared=False)

    ccc_train = ccc(y_phat_train, y_train)
    ccc_test = ccc(y_phat_test, y_test)

    # Get train roc auc
    train_fpr, train_tpr, _ = roc_curve(y_train > 0.4, y_phat_train)
    train_roc_auc = auc(train_fpr, train_tpr)

    # get test roc auc
    test_fpr, test_tpr, _ = roc_curve(y_test > 0.4, y_phat_test)
    test_roc_auc = auc(test_fpr, test_tpr)

    d_info = {
        'train_rmse': round(float(rmse_train), 3),
        'test_rmse': round(float(rmse_test), 3),
        'train_ccc': round(float(ccc_train), 3),
        'test_ccc': round(float(ccc_test), 3),
        'train_accuracy': round(float(accuracy_score(y_train > 0.4, y_hat_train)), 3),
        'test_accuracy': round(float(accuracy_score(y_test > 0.4, y_hat_test)), 3),
        'train_roc_auc': round(float(train_roc_auc), 3),
        'test_roc_auc': round(float(test_roc_auc), 3)
    }
    if display:
        print(d_info)

    if include_graph:
        return {
            **d_info, 'train_cm': confusion_matrix(y_train > 0.4, y_hat_train),
            'test_cm': confusion_matrix(y_test > 0.4, y_hat_test),
            'train_roc_auc': {'fpr': train_fpr, 'tpr': train_tpr, 'roc_auc': round(float(train_roc_auc), 3)},
            'test_roc_auc': {'fpr': test_fpr, 'tpr': test_tpr, 'roc_auc': round(float(test_roc_auc), 3)}
        }

    return d_info


def init_gs(
        gridsearch_path: Path, model_path: Path, cls_model: Any,
) -> Tuple[float, Union[TfModelFull, TfModelSeq], Dict[str, Any]]:
    if gridsearch_path.exists():
        with open(gridsearch_path, 'r') as handle:
            d_info = yaml.safe_load(handle)

        if model_path.exists():
            best_score = max([v['test_ccc'] for v in d_info.values()])
            best_model = cls_model.load(model_path)
        else:
            best_score, best_model = 0, None
    else:
        best_score, best_model, d_info = 0, None, {}

    return best_score, best_model, d_info


def backup_gs(
        gridsearch_path: Path, model_path: Path, d_info: Dict[str, Any], best_model: Any,
        save_model: bool
) -> None:
    if model_path.exists() and save_model:
        shutil.rmtree(model_path.as_posix())

    with open(gridsearch_path, 'w') as handle:
        yaml.dump(d_info, handle, default_flow_style=False)

    if save_model:
        best_model.save(model_path)


def tf_grid_search(
        cls_model: Callable[[Any, Any], Union[TfModelFull, TfModelSeq]], meta_ds: DatasetMeta, train_ds: Dataset,
        test_ds: Dataset, params: Dict[str, Union[int, float, str]],
        grid_params: Dict[str, List[Union[int, float, str]]], gridsearch_path: Path, model_path: Path
) -> Tuple[Union[TfModelFull, TfModelSeq], Dict[str, Any]]:
    """ Implement a grid search of optimal parameter for a Keras model.

    """
    # Init
    best_score, best_model, d_info = init_gs(gridsearch_path, model_path, cls_model)

    # Get numpy data from tf dataset
    X_train, y_train = dataset2numpy(train_ds)
    X_test, y_test = dataset2numpy(test_ds)

    # start grid search
    for l_params in itertools.product(*[[(k, v) for v in l_v] for k, l_v in grid_params.items()]):
        # Fit and evaluate model
        print(f'Training with {l_params}')
        param_name = ','.join(['='.join(list(map(str, [k, v]))) for k, v in l_params])
        if param_name in list(d_info.keys()):
            print(f'skipping {param_name}')
            continue

        fit_thread = FitThread(
            best_score, "test_ccc", "tf", {**params, **{k: v for k, v in l_params}},
            cls_model, X_train=X_train, y_train=y_train, meta_ds=meta_ds, train_ds=train_ds,
            X_test=X_test, y_test=y_test
        )
        fit_thread.start()
        fit_thread.join()

        # Keep best params
        if fit_thread.mdl is not None:
            best_score = fit_thread.best_score
            best_model = fit_thread.mdl

        # Track stats & backup
        if fit_thread.metrics['test_ccc'] > 0.005:
            d_info[param_name] = fit_thread.metrics
            backup_gs(gridsearch_path, model_path, d_info, best_model, save_model=fit_thread.mdl is not None)

        del fit_thread
        tf.keras.backend.clear_session()
        gc.collect()

    return best_model, d_info


def xgb_grid_search(
        cls_model: Callable[[Any, Any, Any], XGBRegressor], train_ds: Dataset, test_ds: Dataset,
        params: Dict[str, Union[int, float, str]], grid_params: Dict[str, List[Union[int, float, str]]],
        gridsearch_path: Path, model_path: Path

) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """Implement a grid search of optimal parameter for a XGB model.

    """
    # Init
    best_score, best_model, d_info = init_gs(gridsearch_path, model_path, cls_model)

    # Get numpy data from tf dataset
    X_train, y_train = dataset2numpy(train_ds)
    X_test, y_test = dataset2numpy(test_ds)

    for l_params in itertools.product(*[[(k, v) for v in l_v] for k, l_v in grid_params.items()]):
        # Fit and evaluate model
        print(f'Training with {l_params}')
        param_name = ','.join(['='.join(list(map(str, [k, v]))) for k, v in l_params])
        if param_name in list(d_info.keys()):
            continue

        fit_thread = FitThread(
            best_score, "test_ccc", "xgb", {**params, **{k: v for k, v in l_params}},
            cls_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        fit_thread.start()
        fit_thread.join()

        # Keep best params
        if fit_thread.mdl is not None:
            best_score = fit_thread.best_score
            best_model = fit_thread.mdl

        # Track stats & backup
        if fit_thread.metrics['test_ccc'] > 0.005:
            d_info[param_name] = fit_thread.metrics
            backup_gs(gridsearch_path, model_path, d_info, best_model, save_model=fit_thread.mdl is not None)

        del fit_thread
        gc.collect()

    return best_model, d_info


class FitThread(threading.Thread):

    def __init__(
            self, best_score: float, target_metric: str, model_type: str, params: Any, cls_model: Any,
            X_train: np.ndarray = None, y_train: np.ndarray = None,
            meta_ds: DatasetMeta = None, train_ds: Dataset = None,
            X_test: np.ndarray = None, y_test: np.ndarray = None,
    ):
        threading.Thread.__init__(self)

        self.model_type = model_type
        self.params = params
        self.cls_model = cls_model
        self.best_score = best_score
        self.target_metric = target_metric

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
            self.metrics = evaluate(mdl, self.X_train, self.y_train, self.X_test, self.y_test)
        else:
            mdl = self.cls_model(self.meta_ds, **self.params) \
                .fit(self.train_ds)
            self.metrics = evaluate(mdl, self.X_train, self.y_train, self.X_test, self.y_test)

        test_perf = self.metrics[self.target_metric]
        train_perf = self.metrics[self.target_metric.replace('test', 'train')]

        if self.best_score < test_perf and test_perf > 0.005 and train_perf >= (test_perf / 3):
            self.mdl = mdl
            self.best_score = test_perf
