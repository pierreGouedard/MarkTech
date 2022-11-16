""" Build features for training.

for each financial time series, split signal according to time. At every instan in
[t0, t0 + predict_interval, ..., t0 + N * predict_interval] we compute the features from data values that took the time
series prior to that instant.

For each instant, the features computed are:
  ### full backward features
   -> deep representation of Gramian of past values of time series up to backward_interval (key=full_gram)
   -> deep representation of MTF of past values of time series up to backward_interval (key=full_mark)
   -> deep representation of ScatterT of past values of time series up to backward_interval (key=full_scat_[ordr])

  ### Segmented backward interval to produce sequence of features (controlled by params window_length & hop_size)
  => deep representation of Gramian of past values of each past segment
   -> Stored as sequence (2D) (key=seq_gram)
   -> Stored as mean of (1D) (key=avg_gram)

  => deep representation of MTF of each past segment
   -> Stored as sequence (2D) (key=seq_mark)
   -> Stored as mean of (1D) (key=avg_mark)

  => deep representation of ScatterT of past values of each past segment
   -> Stored as sequence (2D) (key=seq_scat_[ordr])
   -> Stored as mean of (1D) (key=avg_scat_[ordr])

Store the result of processing in 02 - features.

"""
# Global import
from typing import List, Dict
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from numpy import save, load
from multiprocessing import Pool
from tensorflow.python.keras import Model

project_path = Path(__file__).parent.parent
sys.path.append(project_path.as_posix())

# Local import
from marktech.utils.temp import Temp
from marktech.utils.dataclasses import FeatureMeta
from marktech.time_series.driver import TimeSeriesDriver
from marktech.models.vgg import create_vgg_model, transform_image
from marktech.time_series.features import compute_gramian_angular_field, compute_markov_transition_field, \
    compute_scatter_feature, save_as_image


def init_pool_worker(tmp_dir, image_size, J, Q):

    # Declare scope of a new global variable
    global shared_tmp_dir
    global shared_image_size
    global shared_J
    global shared_Q

    # Store argument in the global variable for this process
    shared_tmp_dir = tmp_dir
    shared_image_size = image_size
    shared_J = J
    shared_Q = Q


def compute_features(ts: TimeSeriesDriver, key: str):

    # Compute representation of
    ts_gram = compute_gramian_angular_field(ts.data.values, shared_image_size)
    ts_mark = compute_markov_transition_field(ts.data.values, shared_image_size)
    l_ts_scatter = compute_scatter_feature(ts.data.values, shared_J, shared_Q)

    # Save signals
    save_as_image(shared_tmp_dir.path / f'{key}_gram.png', ts_gram)
    save_as_image(shared_tmp_dir.path / f'{key}_mark.png', ts_mark)
    if key == 'full':
        save((shared_tmp_dir.path / f'{key}_scat_0.npy').as_posix(), l_ts_scatter[0])
        save_as_image(shared_tmp_dir.path / f'{key}_scat_1.png', l_ts_scatter[1])
        save_as_image(shared_tmp_dir.path / f'{key}_scat_2.png', l_ts_scatter[2])
    else:
        save((shared_tmp_dir.path / f'{key}_scat_1.npy').as_posix(), l_ts_scatter[1][:, 0])
        save((shared_tmp_dir.path / f'{key}_scat_2.npy').as_posix(), l_ts_scatter[2][:, 0])

    return True


def seq_compute_features(ts: TimeSeriesDriver, key: str, image_size: int, pth: Path, J: int, Q: int):

    # Compute representation of
    l_ts_scatter = compute_scatter_feature(ts.data.values, J, Q)
    ts_gram = compute_gramian_angular_field(ts.data.values, image_size)
    ts_mark = compute_markov_transition_field(ts.data.values, image_size)

    # Save signals
    save_as_image(pth / f'{key}_gram.png', ts_gram)
    save_as_image(pth / f'{key}_mark.png', ts_mark)
    if key == 'full':
        save((pth / f'{key}_scat_0.npy').as_posix(), l_ts_scatter[0])
        save_as_image(pth / f'{key}_scat_1.png', l_ts_scatter[1])
        save_as_image(pth / f'{key}_scat_2.png', l_ts_scatter[2])
    else:
        save((pth / f'{key}_scat_1.npy').as_posix(), l_ts_scatter[1][:, 0])
        save((pth / f'{key}_scat_2.npy').as_posix(), l_ts_scatter[1][:, 0])

    return True


def move_final_feature(vgg_model: Model, tmp_path: Path, dest_path: Path, n_frame: int) -> None:

    # Full backward interval features
    for k in ['gram', 'mark', 'scat_1', 'scat_2']:
        # Transform full backward interval features
        ax_vgg = transform_image(tmp_path / f'full_{k}.png', vgg_model, is_dir=False)
        save((dest_path / f'full_{k}.npy').as_posix(), ax_vgg)

    # scatterT order 0 (no deep representation)
    ax_full_scat0 = load((tmp_path / f'full_scat_0.npy').as_posix())
    save((dest_path / f'full_scat_0.npy').as_posix(), ax_full_scat0)

    # Backward sequence of features
    for k in ['gram', 'mark']:
        ax_vgg = transform_image(tmp_path, vgg_model, n_frame=n_frame, regex=r"([0-9]{1,3}\_" + rf"{k}" + r"\.png)")
        save((dest_path / f'seq_{k}.npy').as_posix(), ax_vgg)

    # ScattT tranform:
    ax_seq_scat1 = np.stack([load((tmp_path / f'{i}_scat_1.npy').as_posix()) for i in range(n_frame)])
    ax_seq_scat2 = np.stack([load((tmp_path / f'{i}_scat_2.npy').as_posix()) for i in range(n_frame)])
    save((dest_path / f'seq_scat_1.npy').as_posix(), ax_seq_scat1)
    save((dest_path / f'seq_scat_2.npy').as_posix(), ax_seq_scat2)


if __name__ == '__main__':
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    raw_data_path = data_path / "01 - raw"
    pth_feature_out = data_path / "02 - features"

    # Parameters
    sampling_rate = 1 / 60  # sampled every 60s
    predict_interval = 60  # 60 minutes of existing
    backward_interval = 1026  # ~17H of existing data (~ must be a power of 2 for scatter transform)
    windows_length = 64  # 60 minutes of existing data (~ must be a power of 2 for scatter transform)
    hop_size = 64  # 60 minutes of existing data (~ must be a power of 2 for scatter transform)
    n_frame = 16  # 24H / 60 minutes
    image_size = 224  # VGG required image size
    tresh_date_test = pd.Timestamp('2022-10-01 00:00:00').tz_localize('utc')
    J, Q = 6, 10  # Parameters of scatter transform (randomly guess)

    # Init var
    vgg_model = create_vgg_model()
    l_meta = []

    for filename in os.listdir(raw_data_path):

        # If already exists, continue otherwise, create feature dir
        key = str(filename).split('.')[0]
        pth_sub_out = pth_feature_out / key
        if pth_sub_out.exists():
            continue

        # Create dir and log
        pth_sub_out.mkdir()
        print(f'Computing features for {key}')

        # Read audio
        ts_data = (
            TimeSeriesDriver(time_col='Local time', value_col='Open', sr=sampling_rate)
            .read(raw_data_path / str(filename))
        )

        # Split file to create each sample
        seg_inds = np.arange(backward_interval, len(ts_data.data), predict_interval)
        l_segments = [
            (ts_data.segment_ind(t - backward_interval, t), str(i)) for i, t in enumerate(seg_inds)
        ]

        for (segment, sample_id) in l_segments:
            pth_ssub_out = pth_sub_out / str(sample_id)
            if pth_ssub_out.exists():
                continue

            # Create dir and log
            pth_ssub_out.mkdir()
            print(f'Computing features for {key} - sample {sample_id}')

            # Add meta about sample
            meta = FeatureMeta(
                key=key,
                id=int(sample_id),
                date_start=str(segment.data.index.min()),
                date_end=str(segment.data.index.max()),
                is_test=segment.data.index.max() > tresh_date_test
            )
            l_meta.append(meta.__dict__)

            # Create tmp dir
            tmp_dir = Temp(
                prefix=f'marktech_{key}_{sample_id}_', suffix='_ftrs', is_dir=True,
                dir=(data_path / '06 - tmp').as_posix()
            )

            # segment sample
            seg_sub_inds = np.arange(windows_length, len(segment.data), windows_length)
            l_sub_segments = [(segment, 'full')] + [
                (segment.segment_ind(t - windows_length, t), str(j))
                for j, t in enumerate(seg_sub_inds)
            ]

            # Compute all features for that sample.
            with Pool(4, initializer=init_pool_worker, initargs=(tmp_dir, image_size, J, Q)) as p:
                l_res = list(p.starmap(compute_features, l_sub_segments))

            # Save final features
            move_final_feature(vgg_model, tmp_dir.path, pth_ssub_out, n_frame)

            # Save final vgg features
            tmp_dir.remove()
