# utils module
from typing import List, Callable
from pathlib import Path
import threading
import glob
import os

# Scientific module
import tensorflow as tf
import pandas as pd
import numpy as np
import gc

# Local import
from marktech.utils.dataclasses import DatasetMeta


class PrepareDatasetThread(threading.Thread):

    def __init__(
            self, l_dirs: List[str], load_func: Callable, df_meta: pd.DataFrame, df_labels: pd.DataFrame,
            path_features: Path, key: str
    ):
        threading.Thread.__init__(self)
        self.dirs = l_dirs
        self.load_fn = load_func
        self.raw_meta = df_meta
        self.labels = df_labels
        self.path = path_features
        self.key = key
        self.res = None

    def run(self) -> None:
        # Create Meta dataset
        ds_meta = DatasetMeta().init_mapping()

        # Create train and test empty slices
        l_train_feat, l_train_labels, n_train = [], [], 0
        l_test_feat, l_test_labels, n_test = [], [], 0

        #
        for ids in list(glob.glob((self.path / '*/*').as_posix())):
            [stk_key, sid] = Path(ids).relative_to(self.path).as_posix().split('/')

            # Get from raw meta whether this is a test sample or not
            is_test = self.raw_meta.loc[
                (self.raw_meta.key == stk_key) & (self.raw_meta.id == int(sid)),
                'is_test'
            ].all()
            label = self.labels.loc[
                (self.labels.key == stk_key) & (self.labels.id == int(sid)),
                'target'
            ].iloc[0]

            if not is_test:
                l_train_feat.append(self.load_fn(self.path / stk_key / sid / self.key))
                l_train_labels.append(label)
                ds_meta.train_tf2id[n_train] = {"key": stk_key, "id": sid}
                ds_meta.train_id2tf[f'{stk_key}_{sid}'] = n_train
                n_train += 1

            else:
                l_test_feat.append(self.load_fn(self.path / stk_key / sid / self.key))
                l_test_labels.append(label)
                ds_meta.test_tf2id[n_test] = {"key": stk_key, "id": sid}
                ds_meta.test_id2tf[f'{stk_key}_{sid}'] = n_train

                n_test += 1

        self.res = {
            "train": (np.array(l_train_feat), np.array(l_train_labels, dtype=float)),
            "test": (np.array(l_test_feat), np.array(l_test_labels, dtype=float)),
            "ds_meta": ds_meta
        }


def generic_build_ds(
        pth_features: Path, save_path: Path, df_meta: pd.DataFrame, df_labels: pd.DataFrame,
        key: str, load_fn: Callable, use_normalisation: bool = False
) -> DatasetMeta:

    # List dirs and shuffle
    l_dirs = os.listdir(pth_features)

    # Launch preparation of dataset
    prepare_thread = PrepareDatasetThread(l_dirs, load_fn, df_meta, df_labels, pth_features, key)
    prepare_thread.start()
    prepare_thread.join()

    ax_train_feat, ax_train_labels = prepare_thread.res['train']
    ax_test_feat, ax_test_labels = prepare_thread.res['test']
    ds_meta = prepare_thread.res['ds_meta']

    # Build datasets
    n_samples = ax_train_feat.shape[0]
    ds_meta.val_before = int(n_samples * 0.05)

    if use_normalisation:
        ds_meta.norm_data = np.stack(ax_train_feat[ds_meta.val_before:])

    # Build and save train dataset
    train_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(ax_train_feat),
        tf.data.Dataset.from_tensor_slices(ax_train_labels),
    ))
    train_dataset.save((save_path / 'train_dataset').as_posix())

    # Free memory
    del ax_train_feat
    del ax_train_labels
    del train_dataset
    gc.collect()

    # Build and save test dataset
    test_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(ax_test_feat),
        tf.data.Dataset.from_tensor_slices(ax_test_labels),
    ))
    test_dataset.save((save_path / 'test_dataset').as_posix())
    # Free memory
    del ax_test_feat
    del ax_test_labels
    del test_dataset
    gc.collect()

    return ds_meta


def build_full_datasets(
        pth_features: Path, save_path: Path, df_meta: pd.DataFrame, df_labels: pd.DataFrame,
        key: str, dim: int
) -> None:

    def load(pth: Path):
        ax_feat = np.load(pth.as_posix())
        if len(ax_feat.shape) > 1:
            return ax_feat[0]
        return ax_feat

    save_path.mkdir()
    meta_ds = generic_build_ds(
        pth_features, save_path, df_meta, df_labels, key=f'full_{key}.npy', load_fn=load,
        use_normalisation=True
    )
    # Save meta
    meta_ds.input_dim = (dim,)
    meta_ds.save(save_path / 'meta')


def build_seq_datasets(
        pth_features: Path, save_path: Path, df_meta: pd.DataFrame, df_labels: pd.DataFrame,
        key: str, n_frame: int, dim: int
) -> None:

    def load(pth: Path):
        return np.load(pth.as_posix())

    meta_ds = generic_build_ds(
        pth_features, save_path, df_meta, df_labels, key=f'seq_{key}.npy', load_fn=load,
        use_normalisation=True
    )
    # Save meta
    meta_ds.input_dim = (n_frame, dim)
    meta_ds.save(save_path / 'meta')
