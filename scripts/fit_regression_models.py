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
from marktech.models.tf_conv import TfModelConv
from marktech.models.tf_gru import TfModelGRU
from marktech.models.xgb_full import XGBRegressor
from marktech.search.grid_search import grid_search
from marktech.search.hp_search import run_hp_search


class FittingArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="Arg parser for fitting models.")
        self.add_argument(
            "--dataset",
            metavar="dataset",
            default="seq_gram",
            type=str,
            help="Name of dataset to use for training: {full_gram, seq_gram, full_mark, seq_mark, full_scat_1, "
                 "seq_scat_1, full_scat_2, seq_scat_2, full_scat_0}",
        )
        self.add_argument(
            "--model",
            metavar="model",
            default="conv",
            type=str,
            help="Name of model to use for training: {conv, fc, gru, xgb}",
        )
        self.add_argument(
            "--search",
            metavar="search",
            default="hpsearch",
            type=str,
            help="Name of search to perform: {hpsearch, gridsearch}",
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
    search_path = data_path / "05 - search"

    # Parse args
    args = FittingArgParser().parse_args()

    # Parameters
    weights = (1, 0.5)

    # Get dataset
    if 'full' in args.dataset:
        ds_train, ds_test, ds_meta = load_training_datasets(dataset_path / args.dataset)
        param_file = "gs_tf_full_params.yaml" if args.model == 'fc' else "gs_xgb_full_params.yaml"
        d_params = yaml.safe_load((param_path / param_file).open())
        mdl = TfModelFull if args.model == 'fc' else XGBRegressor

    elif 'seq' in args.dataset:
        ds_train, ds_test, ds_meta = load_training_datasets(dataset_path / args.dataset)
        param_file = "gs_tf_conv_params.yaml" if args.model == 'conv' else "gs_tf_gru_params.yaml"
        d_params = yaml.safe_load((param_path / param_file).open())
        mdl = TfModelConv if args.model == 'conv' else TfModelGRU

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    if args.model not in ['conv', 'gru']:
        if 'seq' in args.dataset:
            raise ValueError('cannot use Xgboost for sequence dataset')

    if args.search == "gridsearch":
        gridsearch_path = search_path / 'grid'
        if not gridsearch_path.exists():
            gridsearch_path.mkdir()

        best_model, d_info = grid_search(
            mdl, ds_meta, ds_train, ds_test, weights, d_params['params'], d_params['grid_params'],
            gridsearch_path / f'gs_{args.model}_{args.dataset}.yaml', model_path / f'{args.model}_{args.dataset}'
        )
    elif args.search == 'hpsearch':
        hpsearch_path = search_path / 'hp'
        if not hpsearch_path.exists():
            hpsearch_path.mkdir()

        trials, best_params = run_hp_search(
            mdl, ds_meta, ds_train, ds_test, weights, d_params['params'], d_params['grid_params'],
            hpsearch_path / f'gs_{args.model}_{args.dataset}.yaml'
        )
