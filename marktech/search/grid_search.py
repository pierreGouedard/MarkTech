# Global import
from typing import List, Dict, Tuple, Union, Callable, Any
from tensorflow.python.data import Dataset
from pathlib import Path
import tensorflow as tf
import itertools
import gc

# local import
from marktech.search.utils import dataset2numpy, init_search, backup_search, FitThread
from marktech.models.datasets import DatasetMeta
from marktech.models.tf_conv import TfModelConv
from marktech.models.tf_full import TfModelFull
from marktech.models.tf_gru import TfModelGRU


def grid_search(
        cls_model: Callable[[Any, Any], Any], meta_ds: DatasetMeta, train_ds: Dataset,
        test_ds: Dataset, weights: Tuple[float, float], params: Dict[str, Union[int, float, str]],
        grid_params: Dict[str, List[Union[int, float, str]]], gridsearch_path: Path, model_path: Path
) -> Tuple[Any, Dict[str, Any]]:
    """ Implement a brut force grid search of optimal parameter of model.

    """
    # Init
    best_score, best_model, d_info = init_search('test_wmse', gridsearch_path, model_path, cls_model, 1e10)

    # Get numpy data from tf dataset
    X_train, y_train = dataset2numpy(train_ds)
    X_test, y_test = dataset2numpy(test_ds)

    # Determine model type
    model_type = 'tf' if cls_model in [TfModelGRU, TfModelConv, TfModelFull] else 'xgb'

    # start grid search
    for l_params in itertools.product(*[[(k, v) for v in l_v] for k, l_v in grid_params.items()]):
        # Fit and evaluate model
        print(f'Training with {l_params}')
        param_name = ','.join(['='.join(list(map(str, [k, v]))) for k, v in sorted(l_params, key=lambda t: t[0])])
        if param_name in list(d_info.keys()):
            print(f'skipping {param_name}')
            continue

        fit_thread = FitThread(
            best_score, "test_wmse", model_type, {**params, **{k: v for k, v in l_params}},
            cls_model, weights, X_train=X_train, y_train=y_train, meta_ds=meta_ds, train_ds=train_ds,
            X_test=X_test, y_test=y_test
        )
        fit_thread.run()
        #fit_thread.start()
        #fit_thread.join()

        # Keep best params
        if fit_thread.mdl is not None:
            best_score = fit_thread.best_metric
            best_model = fit_thread.mdl

        # Track stats & backup
        d_info[param_name] = fit_thread.metrics
        backup_search(gridsearch_path, d_info, model_path, best_model, save_model=fit_thread.mdl is not None)

        del fit_thread
        tf.keras.backend.clear_session()
        gc.collect()

    return best_model, d_info
