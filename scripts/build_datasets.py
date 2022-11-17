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
from marktech.models.datasets import build_full_datasets, build_seq_datasets


if __name__ == '__main__':
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    raw_metadata_path = data_path / "01 - raw" / "metadata"
    features_path = data_path / "02 - features"
    dataset_path = data_path / "03 - datasets"

    # Parameters
    vgg_size = 4096
    scat_0_size = 17
    scat_1_size = 45
    scat_2_size = 106
    forward_interval = 1026
    label_name = 'next_max'
    n_frame = 16

    # load feature metadata:
    df_meta = pd.read_csv(features_path / "meta.csv", index_col=None)
    df_labels = (
        pd.read_csv(features_path / "labels.csv", index_col=None)
        .loc[:, ['key', 'id', label_name]]
        .rename(columns={label_name: 'target'})
    )

    # Build full datasets
    l_full_features = [
        ('gram', vgg_size), ('mark', vgg_size), ('scat_0', scat_0_size), ('scat_1', vgg_size),
        ('scat_2', vgg_size),
    ]
    for (key, dim) in l_full_features:
        build_full_datasets(features_path, dataset_path / f'full_{key}', df_meta, df_labels, key=key, dim=dim)

    # Build seq datasets
    l_seq_features = [
        ('gram', vgg_size), ('mark', vgg_size), ('scat_1', scat_1_size), ('scat_2', scat_2_size),
    ]
    for (key, dim) in l_seq_features:
        build_seq_datasets(
            features_path, dataset_path / f'seq_{key}', df_meta, df_labels, key=key, dim=dim, n_frame=n_frame
        )
