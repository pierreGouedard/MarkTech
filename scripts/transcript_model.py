""" Modelise PHQ-8 scores from text transcript of interview.

This script implements a modelisation of PHQ-8 scores using exclusively the transcript of conversation as raw data. The
transcript is then passed through a sentence transformer to do the regression of PHQ-8 scores. Although this method
doesn't address the problem of the acquisition of transcript from audio recording, it shows promising results.

"""
# Global import
from typing import Dict, Any
from pathlib import Path
import yaml
import sys
import os

project_path = Path(__file__).parent.parent
sys.path.append(project_path.as_posix())

# Computation module
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import numpy as np
import pandas as pd

# Local model import
from marktech.models.tf_seq import TfModelSeq
from marktech.models.xgb_full import XGBRegressor
from marktech.search.grid_search import tf_grid_search, xgb_grid_search


# Local util import
from marktech.models.datasets import DatasetMeta

# Define paths
data_path = project_path / "data"
raw_data_path = data_path / "01 - raw"
raw_metadata_path = data_path / "01 - raw" / "metadata"
param_path = project_path / "config" / "global"
model_path = data_path / "04 - models"
gridsearch_path = data_path / "05 - gridsearch"


def build_embeddings_dataset(
        d_embeddings: Dict[int, Any], s_train_labels: pd.Series, s_test_labels: pd.Series,
        n_frame: int, embeddings_dim: int
):
    # init dataset meta
    ds_meta = DatasetMeta().init_mapping()

    # Create train and test slices
    l_train_feat_full, l_train_feat_seq, l_train_labels, n_train = [], [], [], 0
    l_test_feat_full, l_test_feat_seq, l_test_labels, n_test = [], [], [], 0

    # DO it with multip
    for i, (sample_id, ax_embeddings) in enumerate(d_embeddings.items()):

        if sample_id in s_train_labels.index:
            l_train_feat_full.append(ax_embeddings.mean(axis=0))
            l_train_feat_seq.append(ax_embeddings)
            l_train_labels.append(s_train_labels[sample_id])
            ds_meta.train_ind_id2tf[sample_id] = n_train
            ds_meta.train_ind_tf2id[n_train] = sample_id
            n_train += 1

        elif sample_id in s_test_labels.index:
            l_test_feat_full.append(ax_embeddings.mean(axis=0))
            l_test_feat_seq.append(ax_embeddings)
            l_test_labels.append(s_test_labels[sample_id])
            ds_meta.test_ind_id2tf[sample_id] = n_test
            ds_meta.test_ind_tf2id[n_test] = sample_id
            n_test += 1

    # Build datasets
    n_samples = len(l_train_feat_full)
    ds_meta.val_before = int(n_samples * 0.05)

    # Build and save train dataset
    train_full_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(np.array(l_train_feat_full)),
        tf.data.Dataset.from_tensor_slices(np.array(l_train_labels)),
    ))

    # Build and save test dataset
    test_full_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(np.array(l_test_feat_full)),
        tf.data.Dataset.from_tensor_slices(np.array(l_test_labels)),
    ))

    train_seq_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(np.array(l_train_feat_seq)),
        tf.data.Dataset.from_tensor_slices(np.array(l_train_labels)),
    ))

    # Build and save test dataset
    test_seq_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(np.array(l_test_feat_seq)),
        tf.data.Dataset.from_tensor_slices(np.array(l_test_labels)),
    ))

    # Set Metadata for seq dataset
    ds_meta.input_dim = (n_frame, embeddings_dim)

    return train_full_dataset, test_full_dataset, train_seq_dataset, test_seq_dataset, ds_meta


if __name__ == '__main__':

    transformer_model = "sentence-transformers/all-MiniLM-L6-v2"
    sntnce_model = SentenceTransformer(transformer_model)
    n_frame = 150
    embeddings_dim = 384
    downscale = 25

    # I.features
    d_embeddings = {}
    for filename in os.listdir(raw_data_path):
        if filename == 'metadata':
            continue

        sample_id = int(str(filename).split("_")[0])
        transcript_path = raw_data_path / str(filename) / f'{sample_id}_Transcript.csv'
        if transcript_path.exists():
            df_transcript = pd.read_csv(transcript_path)
            ax_embeddings = sntnce_model.encode(list(df_transcript['Text'].values[:n_frame]))

            if ax_embeddings.shape[0] < n_frame:
                ax_embeddings = np.concatenate([
                    ax_embeddings,
                    np.zeros((n_frame - ax_embeddings.shape[0], ax_embeddings.shape[-1]))
                ])
            d_embeddings[sample_id] = ax_embeddings

    # II. build datasets
    df_train_split = pd.read_csv(raw_metadata_path / "train_split.csv")
    s_train_lbl = df_train_split.set_index('Participant_ID')['PHQ_Score'].astype(float) / downscale
    df_test_split = pd.read_csv(raw_metadata_path / "dev_split.csv")
    s_test_lbl = df_test_split.set_index('Participant_ID')['PHQ_Score'].astype(float) / downscale
    (
        ds_train_full, ds_test_full, ds_train_seq, ds_test_seq, ds_meta_seq
    ) = build_embeddings_dataset(d_embeddings, s_train_lbl, s_test_lbl, n_frame, embeddings_dim)

    # III. train model
    #   A. xgb model for full
    d_params = yaml.safe_load((param_path / "gs_xgb_full_params.yaml").open())
    best_model, d_info = xgb_grid_search(
        XGBRegressor, ds_train_full, ds_test_full, d_params['params'], d_params['grid_params'],
        gridsearch_path / f'gs_transcript_full_model.yaml', model_path / f'gs_transcript_full_model'
    )

    #   B. TfSeq model for sequence dataset.
    d_params = yaml.safe_load((param_path / "gs_tf_seq_params.yaml").open())
    best_seq_model, d_seq_info = tf_grid_search(
        TfModelSeq, ds_meta_seq, ds_train_seq, ds_test_seq, d_params['params'], d_params['grid_params'],
        gridsearch_path / 'gs_transcript_seq_model.yaml', model_path / 'gs_transcript_seq_model'
    )

