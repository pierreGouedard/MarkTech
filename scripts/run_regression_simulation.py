"""Run simulation

"""
# Utils import
from typing import Tuple, List
from pathlib import Path
import argparse
import yaml
import sys

project_path = Path(__file__).parent.parent
sys.path.append(project_path.as_posix())

# computation import
from tensorflow.python.data import Dataset

# Local import
from marktech.models.datasets import DatasetMeta
from marktech.models.tf_full import TfModelFull
from marktech.models.tf_conv import TfModelConv
from marktech.models.tf_gru import TfModelGRU
from marktech.models.xgb_full import XGBRegressor
from marktech.utils.dataclasses import FeatureMeta, Labels
from marktech.time_series.driver import TimeSeriesDriver


class SimulationArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="Arg parser for Simulation.")
        self.add_argument(
            "--model",
            metavar="model",
            default="tf",
            type=str,
            help="Name of model to use for training: {bagging, tf_full_gram, xgb_full_gram, tf_seq_gram, gru_seq_gram, "
                 "tf_full_mark, xgb_full_mark, tf_seq_mark, gru_seq_mark, tf_full_scat_1, xgb_full_scat_1, "
                 "tf_seq_scat_1, gru_seq_scat_1, tf_full_scat_2, xgb_full_scat_2, tf_seq_scat_2, gru_seq_scat_2, "
                 "tf_full_scat_0, xgb_full_scat_0}",
        )


def load_data(dataset_path: Path, model_path: Path, l_mdl_keys: List[str]) -> Tuple[Dataset, DatasetMeta]:
    for mdl_key in l_mdl_keys:
        ds_key = '_'.join(mdl_key.split('_')[1:])
        ds_test = Dataset.load((dataset_path / ds_key / 'test_dataset').as_posix())
        ds_meta = DatasetMeta().load(dataset_path / ds_key / "meta")
        mdl = d_all_models[mdl_key].load(model_path)

    return ds_test, ds_meta



d_all_models = {
    # Gramian based
    "tf_full_gram": TfModelFull,
    "xgb_full_gram": XGBRegressor,
    "tf_seq_gram": TfModelConv,
    "gru_seq_gram": TfModelGRU,

    # Markov based
    "tf_full_mark": TfModelFull,
    "xgb_full_mark": XGBRegressor,
    "tf_seq_mark": TfModelConv,
    "gru_seq_mark": TfModelGRU,

    # Scatter 1
    "tf_full_scat_1": TfModelFull,
    "xgb_full_scat_1": XGBRegressor,
    "tf_seq_scat_1": TfModelConv,
    "gru_seq_scat_1": TfModelGRU,

    # Scatter 2
    "tf_full_scat_2": TfModelFull,
    "xgb_full_scat_2": XGBRegressor,
    "tf_seq_scat_2": TfModelConv,
    "gru_seq_scat_2": TfModelGRU,

    # Scatter 0
    "tf_full_scat_0": TfModelConv,
    "xgb_full_scat_0": TfModelConv,
}

if __name__ == '__main__':
    # Get project path
    project_path = Path(__file__).parent.parent

    data_path = project_path / "data"
    raw_path = project_path / "01 - raw"
    dataset_path = data_path / "03 - datasets"
    model_path = data_path / "04 - models"

    # Parameters
    allowed_index = ['AC', 'ALO', 'BN', 'BNP', 'DSY', 'ENGI', 'FP', 'HO', 'RNO', 'SAF', 'VK']
    sampling_rate = 1 / 60

    # Parse args
    args = SimulationArgParser().parse_args()

    # Load model
    if d_all_models.get(args.model, None) is not None:
        # Load specific dataset and specific model
        pass

    elif args.model == 'bagging':
        # Load all model and all datasets
        pass
    else:
        raise ValueError(f'model name {args.model} not allowed')

    d_ts_data = {}
    for k in allowed_index:
        d_ts_data[k] = (
            TimeSeriesDriver(time_col='Local time', value_col='Open', sr=sampling_rate)
            .read(raw_path / f'{k}.csv')
        )

