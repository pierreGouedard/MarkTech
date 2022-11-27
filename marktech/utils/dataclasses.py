from typing import Dict, List, Any
from dataclasses import dataclass
from copy import deepcopy as copy
from pathlib import Path
import numpy as np
import json


@dataclass
class FeatureMeta:
    key: str
    id: int
    date_start: str
    date_end: str
    is_test: bool


@dataclass
class Labels:
    key: str
    id: int
    is_up: bool
    is_up_1: bool
    is_up_2: bool
    is_up_3: bool
    is_up_4: bool
    is_up_5: bool
    is_up_6: bool
    is_up_7: bool
    is_up_8: bool
    is_up_9: bool
    is_up_10: bool
    is_down: bool
    is_down_1: bool
    is_down_2: bool
    is_down_3: bool
    is_down_4: bool
    is_down_5: bool
    is_down_6: bool
    is_down_7: bool
    is_down_8: bool
    is_down_9: bool
    is_down_10: bool
    next_max: float
    next_max_relative: float
    next_min: float
    next_min_relative: float


@dataclass
class DatasetMeta:
    train_tf2id: Dict[int, Dict[str, str]] = None
    train_id2tf: Dict[str, int] = None
    test_tf2id: Dict[int, Dict[str, str]] = None
    test_id2tf: Dict[str, int] = None
    val_before: int = None
    norm_data: np.ndarray = None
    input_dim: Any = None

    def copy(self):
        return DatasetMeta(
            copy(self.train_tf2id), copy(self.train_id2tf), copy(self.test_tf2id), copy(self.test_id2tf),
            self.val_before, self.norm_data, copy(self.input_dim)
        )

    def init_mapping(self):
        self.train_tf2id = {}
        self.train_id2tf = {}
        self.test_tf2id = {}
        self.test_id2tf = {}
        return self

    def load(self, path: Path):
        with open(path / 'map.json', 'r') as handle:
            d_map = json.load(handle)

        self.train_tf2id = d_map['train_tf2id']
        self.test_tf2id = d_map['test_tf2id']
        self.train_id2tf = d_map['train_id2tf']
        self.test_id2tf = d_map['test_id2tf']
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
                    "train_tf2id": self.train_tf2id,
                    "train_id2tf": self.train_tf2id,
                    "test_tf2id": self.test_tf2id,
                    "test_id2tf": self.test_tf2id,
                    "val_before": self.val_before,
                    "input_dim": self.input_dim,
                }, handle
            )
        if self.norm_data is not None:
            np.save((path / 'norm_data.npy').as_posix(), self.norm_data)
