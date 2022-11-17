"""Train model with grid search.

The models to train are:
  - Dense Fully connect neural network (tensorflow) from deep representation of spectrogram of full interview
  - XGBOOST model from deep representation of spectrogram of full interview
  - Dense Fully connect neural network (tensorflow) from lld over full interview
  - XGBOOST model from lld over full interview
  - 1D conv neural network model (tensorflow) from sequence of deep representation of spectrogram of fixed length
  - GRU neural network model (tensorflow) from sequence of deep representation of spectrogram of fixed length
  - 1D conv neural network model (tensorflow) from sequence of lld of fixed length
  - GRU neural network model (tensorflow) from sequence of lld of fixed length

The script takes 2 arguments:
  * --dataset: select the features in \[vgg_full, vgg_seq, lld_full, lld_seq\]
  * --model: select the model to train \[tf, gru, xgb\]

The script will raise an error if --dataset arg contain 'seq' and --model is set to 'xgb' since xgboost cannot be chosen
with sequence dataset.

The best model of each model type is stored in data/04 - models. The overview of gridsearch is stored in
data/05 - gridsearch

"""
# Utils import
from typing import Tuple
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
from marktech.models.tf_seq import TfModelSeq
from marktech.models.tf_gru import TfModelGRU
from marktech.models.xgb_full import XGBRegressor
from marktech.utils.grid_search import tf_grid_search, xgb_grid_search


class TrainingArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="Arg parser for Heka task updating.")
        self.add_argument(
            "--dataset",
            metavar="dataset",
            default="vgg_seq",
            type=str,
            help="Name of dataset to use for training: {lld_full, lld_seq, vgg_full, vgg_seq}",
        )
        self.add_argument(
            "--model",
            metavar="model",
            default="tf",
            type=str,
            help="Name of model to use for training: {tf, xgb}",
        )


def load_training_datasets(dataset_path: Path) -> Tuple[Dataset, Dataset, DatasetMeta]:
    ds_train = Dataset.load((dataset_path / 'train_dataset').as_posix())
    ds_test = Dataset.load((dataset_path / 'test_dataset').as_posix())
    ds_meta = DatasetMeta().load(dataset_path / "meta")

    return ds_train, ds_test, ds_meta


if __name__ == '__main__':
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    dataset_path = data_path / "03 - datasets"
    param_path = project_path / "config" / "global"
    model_path = data_path / "04 - models"
    gridsearch_path = data_path / "05 - gridsearch"

    # Parse args
    args = TrainingArgParser().parse_args()

    # Get dataset
    if args.dataset == 'lld_full':
        ds_train, ds_test, ds_meta = load_training_datasets(dataset_path / 'lld_full')
        param_file = "gs_tf_full_params.yaml" if args.model == 'tf' else "gs_xgb_full_params.yaml"
        d_params = yaml.safe_load((param_path / param_file).open())
        mdl = TfModelFull if args.model in ['tf', 'gru'] else XGBRegressor

    elif args.dataset == 'vgg_full':
        ds_train, ds_test, ds_meta = load_training_datasets(dataset_path / 'vgg_full')
        param_file = "gs_tf_full_params.yaml" if args.model == 'tf' else "gs_xgb_full_params.yaml"
        d_params = yaml.safe_load((param_path / param_file).open())
        mdl = TfModelFull if args.model in ['tf', 'gru'] else XGBRegressor

    elif args.dataset == 'lld_seq':
        ds_train, ds_test, ds_meta = load_training_datasets(dataset_path / 'lld_seq')
        param_file = "gs_tf_seq_params.yaml" if args.model == 'tf' else "gs_tf_gru_params.yaml"
        d_params = yaml.safe_load((param_path / param_file).open())
        mdl = TfModelSeq if args.model == 'tf' else TfModelGRU

    elif args.dataset == 'vgg_seq':
        ds_train, ds_test, ds_meta = load_training_datasets(dataset_path / 'vgg_seq')
        param_file = "gs_tf_seq_params.yaml" if args.model == 'tf' else "gs_tf_gru_params.yaml"
        d_params = yaml.safe_load((param_path / param_file).open())
        mdl = TfModelSeq if args.model == 'tf' else TfModelGRU

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    if args.model in ['tf', 'gru']:
        best_model, d_info = tf_grid_search(
            mdl, ds_meta, ds_train, ds_test, d_params['params'], d_params['grid_params'],
            gridsearch_path / f'gs_{args.model}_{args.dataset}.yaml', model_path / f'{args.model}_{args.dataset}'
        )
    else:
        if 'seq' in args.dataset:
            raise ValueError('cannot use Xgboost for sequence dataset')
        best_model, d_info = xgb_grid_search(
            mdl, ds_train, ds_test, d_params['params'], d_params['grid_params'],
            gridsearch_path / f'gs_{args.model}_{args.dataset}.yaml', model_path / f'{args.model}_{args.dataset}'
        )