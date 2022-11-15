# utils module
from typing import Dict, List, Callable, Any
from copy import deepcopy as copy
from dataclasses import dataclass
from pathlib import Path
import threading
import json
import os

# Scientific module
from random import shuffle, seed
import tensorflow as tf
import pandas as pd
import numpy as np
import gc


@dataclass
class DatasetMeta:
    train_ind_tf2id: Dict[int, str] = None
    train_ind_id2tf: Dict[str, int] = None
    test_ind_tf2id: Dict[int, str] = None
    test_ind_id2tf: Dict[str, int] = None
    val_before: int = None
    norm_data: np.ndarray = None
    input_dim: Any = None

    def copy(self):
        return DatasetMeta(
            copy(self.train_ind_tf2id), copy(self.train_ind_id2tf), copy(self.test_ind_tf2id),
            copy(self.test_ind_id2tf), self.val_before, self.norm_data, copy(self.input_dim)
        )

    def init_mapping(self):
        self.train_ind_tf2id, self.train_ind_id2tf = {}, {}
        self.test_ind_tf2id, self.test_ind_id2tf = {}, {}
        return self

    def load(self, path: Path):
        with open(path / 'map.json', 'r') as handle:
            d_map = json.load(handle)

        self.train_ind_tf2id = d_map['train_ind_tf2id']
        self.train_ind_id2tf = d_map['train_ind_id2tf']
        self.test_ind_tf2id = d_map['test_ind_tf2id']
        self.test_ind_id2tf = d_map['test_ind_id2tf']
        self.val_before = d_map['val_before']
        self.input_dim = d_map['input_dim']

        if (path / 'norm_data.npy').exists():
            self.norm_data = np.load((path / 'norm_data.npy').as_posix())

        return self

    def save(self, path: Path):
        path.mkdir()
        with open(path / 'map.json', 'w') as handle:
            json.dump(
                {
                    "train_ind_tf2id": self.train_ind_tf2id,
                    "train_ind_id2tf": self.train_ind_id2tf,
                    "test_ind_tf2id": self.test_ind_tf2id,
                    "test_ind_id2tf": self.test_ind_id2tf,
                    "val_before": self.val_before,
                    "input_dim": self.input_dim,
                }, handle
            )
        if self.norm_data is not None:
            np.save((path / 'norm_data.npy').as_posix(), self.norm_data)


class PrepareDatasetThread(threading.Thread):

    def __init__(
            self, l_dirs: List[str], load_func: Callable, train_labels: pd.Series,
            test_labels: pd.Series, path_features: Path, key: str
    ):
        threading.Thread.__init__(self)
        self.dirs = l_dirs
        self.load_fn = load_func
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.path = path_features
        self.key = key
        self.res = None

    def run(self) -> None:
        # Create Meta dataset
        ds_meta = DatasetMeta().init_mapping()

        # Create train and test slices
        l_train_feat, l_train_labels, n_train = [], [], 0
        l_test_feat, l_test_labels, n_test = [], [], 0

        # DO it with multip
        for i, filename in enumerate(self.dirs):
            if int(filename) in self.train_labels.index:
                l_train_feat.append(self.load_fn(self.path / str(filename) / self.key))
                l_train_labels.append(self.train_labels[int(filename)])
                ds_meta.train_ind_id2tf[filename] = n_train
                ds_meta.train_ind_tf2id[n_train] = filename
                n_train += 1

            elif int(filename) in self.test_labels.index:
                l_test_feat.append(self.load_fn(self.path / str(filename) / self.key))
                l_test_labels.append(self.test_labels[int(filename)])
                ds_meta.test_ind_id2tf[filename] = n_test
                ds_meta.test_ind_tf2id[n_test] = filename
                n_test += 1

        self.res = {
            "train": (np.array(l_train_feat), np.array(l_train_labels, dtype=float)),
            "test": (np.array(l_test_feat), np.array(l_test_labels, dtype=float)),
            "ds_meta": ds_meta
        }


def generic_build_ds(
        pth_features: Path, save_path: Path, s_train_labels: pd.Series, s_test_labels: pd.Series,
        key: str, load_fn: Callable, use_normalisation: bool = False
) -> DatasetMeta:

    # List dirs and shuffle
    l_dirs = os.listdir(pth_features)
    seed(123)
    shuffle(l_dirs)

    # Launch preparation of dataset
    prepare_thread = PrepareDatasetThread(
        l_dirs, load_fn, s_train_labels, s_test_labels, pth_features, key
    )
    prepare_thread.start()
    prepare_thread.join()

    ax_train_feat, ax_train_labels = prepare_thread.res['train']
    ax_test_feat, ax_test_labels = prepare_thread.res['test']
    ds_meta = prepare_thread.res['ds_meta']

    # Build datasets
    n_samples = ax_train_feat.shape[0]
    ds_meta.val_before = int(n_samples * 0.1)

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


def build_lld_full_datasets(
        pth_features: Path, save_path: Path, s_train_labels: pd.Series, s_test_labels: pd.Series, lld_dim: int
) -> None:

    def load_llds(pth: Path):
        return np.load(pth.as_posix())

    save_path.mkdir()
    meta_ds = generic_build_ds(
        pth_features, save_path, s_train_labels, s_test_labels,  key='full_lld.npy', load_fn=load_llds,
        use_normalisation=True
    )
    # Save meta
    meta_ds.input_dim = (lld_dim,)
    meta_ds.save(save_path / 'meta')


def build_lld_seq_datasets(
        pth_features: Path, save_path: Path, s_train_labels: pd.Series, s_test_labels: pd.Series, n_frame: int,
        lld_dim: int
) -> None:

    def load_lld_seq(pth: Path):
        return np.load(pth.as_posix())

    meta_ds = generic_build_ds(
        pth_features, save_path, s_train_labels, s_test_labels,  key='seq_lld.npy', load_fn=load_lld_seq,
        use_normalisation=True
    )
    # Save meta
    meta_ds.input_dim = (n_frame, lld_dim)
    meta_ds.save(save_path / 'meta')


def build_vgg_full_datasets(
        pth_features: Path, save_path: Path, s_train_labels: pd.Series, s_test_labels: pd.Series, input_dim: int
) -> None:

    def load_vgg(pth: Path):
        return np.load(pth.as_posix())[0, :]

    meta_ds = generic_build_ds(
        pth_features, save_path, s_train_labels, s_test_labels,  key='full_vgg.npy',
        load_fn=load_vgg, use_normalisation=True
    )

    # Save meta
    meta_ds.input_dim = (input_dim,)
    meta_ds.save(save_path / 'meta')


def build_vgg_seq_datasets(
        pth_features: Path, save_path: Path, s_train_labels: pd.Series, s_test_labels: pd.Series, n_frame: int,
        input_dim: int
) -> None:

    def load_vgg_seq(pth: Path):
        return np.load(pth.as_posix())

    meta_ds = generic_build_ds(
        pth_features, save_path, s_train_labels, s_test_labels,  key='seq_vgg.npy',
        load_fn=load_vgg_seq, use_normalisation=True
    )
    # Save meta
    meta_ds.input_dim = (n_frame, input_dim)
    meta_ds.save(save_path / 'meta')
