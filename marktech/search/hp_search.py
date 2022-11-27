# Utils import
from typing import List, Dict, Tuple, Union, Callable, Any
from pathlib import Path
import gc

# computation import
import tensorflow as tf
from tensorflow.python.data import Dataset
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK

# Local import
from marktech.search.utils import dataset2numpy, init_search, backup_search, FitThread
from marktech.models.datasets import DatasetMeta
from marktech.models.tf_conv import TfModelConv
from marktech.models.tf_full import TfModelFull
from marktech.models.tf_gru import TfModelGRU


class HPObjective:
    def __init__(
            self,  model_type: str, ds_train: Dataset, ds_test: Dataset, meta_ds: DatasetMeta,
            weights: Tuple[float, float], params: Dict[str, Any], hpsearch_path: Path
    ):
        # Algo input
        self.model_type = model_type
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.meta_ds = meta_ds

        # Parameters
        self.weights = weights
        self.params = params

        # Util paths
        self.hpsearch_path = hpsearch_path

        # Backup
        _, _, self.backup = init_search('test_wmse', hpsearch_path)

    def __call__(self, args):

        # Build model args
        mdl_args = {**self.params, **{k: v for k, v in args.items() if k != 'class'}}

        # Search in backup if model as been already trained
        sorted_items = sorted(mdl_args.items(), key=lambda t: t[0])
        param_name = ','.join(['='.join(list(map(str, [k, v]))) for k, v in sorted_items])
        print(f'Training with {param_name}')
        if param_name in list(self.backup.keys()):
            print(f'skipping {param_name}')
            return {**self.backup[param_name], 'loss': self.backup[param_name]['test_wmse'], 'status': STATUS_OK}

        # Get data
        X_train, y_train = dataset2numpy(self.ds_train)
        X_test, y_test = dataset2numpy(self.ds_test)

        # Fit model
        fit_thread = FitThread(
            0., "test_wmse", self.model_type, mdl_args,  args['class'], self.weights,
            X_train=X_train, y_train=y_train, meta_ds=self.meta_ds, train_ds=self.ds_train,
            X_test=X_test, y_test=y_test
        )

        fit_thread.start()
        fit_thread.join()

        # Track stats & backup
        d_info = fit_thread.metrics
        backup_search(self.hpsearch_path, d_info)

        del fit_thread
        tf.keras.backend.clear_session()
        gc.collect()

        return {**d_info, 'loss': d_info['test_wmse'], 'status': STATUS_OK}


def build_space(mdl: Callable, d_params: Dict[str, Any]):
    def to_hp_params(name: str, params: Union[List, Dict]):
        if isinstance(params, list):
            return hp.choice(name, params)
        elif isinstance(params, dict):
            if params['type'] == 'uniform':
                return hp.uniform(name, params['min'], params['max'])
            if params['type'] == 'randint':
                return hp.uniform(name, params['max'])
            if params['type'] == 'loguniform':
                return hp.loguniform(name, params['min'], params['max'])
            if params['type'] == 'normal':
                return hp.normal(name, params['mu'], params['sigma'])
            if params['type'] == 'lognormal':
                return hp.lognormal(name, params['mu'], params['sigma'])

    # Define the search space
    return hp.choice('model', [
        {'class': mdl, **{k: to_hp_params(k, v) for k, v in d_params.items()}}
    ])


def run_hp_search(
        cls_model: Callable[[Any, Any], Any], meta_ds: DatasetMeta, train_ds: Dataset,
        test_ds: Dataset, weights: Tuple[float, float], params: Dict[str, Union[int, float, str]],
        grid_params: Dict[str, List[Union[int, float, str]]], hpsearch_path: Path
) -> Tuple[Trials, Dict[str, Any]]:
    """ Implement an optimized HP search of optimal parameter of model (using hyperopt module).

    """
    # Build search space
    space = build_space(cls_model, grid_params)

    # Create objective callable
    model_type = 'tf' if cls_model in [TfModelGRU, TfModelConv, TfModelFull] else 'xgb'
    objective = HPObjective(model_type, train_ds, test_ds, meta_ds, weights, params, hpsearch_path)

    # Run HP search
    trials = Trials()
    best_params = fmin(
        objective, space=space, algo=tpe.suggest, trials=trials, max_evals=1000
    )

    return trials, best_params


