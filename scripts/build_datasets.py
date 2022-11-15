""" Build training and test datasets for model training & evaluation.

Build the following train & test datasets (tensorflow datasets)
  - full interview deep representation of spectrograms (VGG-16)
  - Sequence of deep representation of spectrograms (VGG-16)
  - full interview llds (baseline)
  - Sequence of llds (baseline)

Each dataset is a valid tensorflow dataset. They can be feeded to any tensorflow model.
In addition, a file of metada for each dataset is saved (dim on inputs, mapping index / sample name, ...)

Store datasets in 03 - datasets

"""
# Global import
from pathlib import Path
import pandas as pd
import sys

project_path = Path(__file__).parent.parent
sys.path.append(project_path.as_posix())

# Local import
from marktech.models.datasets import build_lld_full_datasets, build_lld_seq_datasets, build_vgg_full_datasets, \
    build_vgg_seq_datasets


if __name__ == '__main__':
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    raw_metadata_path = data_path / "01 - raw" / "metadata"
    features_path = data_path / "02 - features"
    dataset_path = data_path / "03 - datasets"

    # Parameters
    vgg_size = 4096
    n_frame = 60
    lld_dim = 133
    downscale = 25

    # load raw metadata:
    df_train_split = pd.concat(
        [pd.read_csv(raw_metadata_path / "train_split.csv"), pd.read_csv(raw_metadata_path / "dev_split.csv")],
        ignore_index=True
    )
    df_test_split = pd.read_csv(raw_metadata_path / "test_split.csv")
    s_train_lbl = df_train_split.set_index('Participant_ID')['PHQ_Score'].astype(float) / downscale
    s_test_lbl = df_test_split.set_index('Participant_ID')['PHQ_Score'].astype(float) / downscale

    # Build lld datasets
    build_lld_full_datasets(
        features_path, dataset_path / 'lld_full', s_train_lbl, s_test_lbl, lld_dim
    )

    build_lld_seq_datasets(
        features_path, dataset_path / 'lld_seq', s_train_lbl, s_test_lbl, n_frame, lld_dim
    )

    # Build vgg datasets
    build_vgg_full_datasets(
        features_path, dataset_path / 'vgg_full', s_train_lbl, s_test_lbl, vgg_size
    )
    build_vgg_seq_datasets(
        features_path, dataset_path / 'vgg_seq', s_train_lbl, s_test_lbl, n_frame, vgg_size
    )
