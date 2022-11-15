""" Build features for training.

For each sample, the features computed are:
    - baseline features (eGeMAPS, mfcc + deltas)
    - deep representation spectrogram images (using imagenet trained VGG16)

there is two ways of computing the feature. The first method consists in computing the feature on the entire interview
length (features stored with name '_full'). the second method is to compute the feature over a widows of a given size,
using a given hop size. The result of the second method consist in a sequence of numeric vectors for each interview (
features stored with name '_seq').

Store the result of processing in 02 - features.

"""
# Global import
import os
import re
import sys
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from numpy import arange, save, load, stack, concatenate, zeros
from librosa.util.exceptions import ParameterError
from tensorflow.python.keras import Model

project_path = Path(__file__).parent.parent
sys.path.append(project_path.as_posix())

# Local import
from marktech.time_series.driver import TimeSeriesDriver
from marktech.utils.temp import Temp
from marktech.models.vgg import create_vgg_model, transform_image
from marktech.audio.features import compute_llds, compute_mel_spectrogram, \
    save_spectrogram_as_image


def init_pool_worker(tmp_dir):
    # declare scope of a new global variable
    global shared_tmp_dir

    # store argument in the global variable for this process
    shared_tmp_dir = tmp_dir


def compute_features(audio: AudioDriver, key: str):

    # Compute audio low level descriptor
    try:
        llds = compute_llds(audio)
        save(shared_tmp_dir.path / f'{key}_lld.npy', llds)
    except ParameterError:
        print(f'Error on lld computation of {shared_tmp_dir.path.name} - {key}')

    # Compute spectrogram
    spec = compute_mel_spectrogram(audio.data, audio.sr)
    save_spectrogram_as_image(shared_tmp_dir.path / f'{key}_spec.png', spec, audio.sr)

    return True


def move_final_lld_feature(tmp_path: Path, dest_path: Path, pattern: str, n_frame: int) -> None:
    # Move full audio lld
    os.rename(tmp_path / "full_lld.npy", dest_path / "full_lld.npy")

    # Build lld sequence and save it to dest
    ax_seq_lld = stack([
        load((tmp_path / f).as_posix()) for f in os.listdir(tmp_path) if re.findall(pattern, str(f))
    ])

    if ax_seq_lld.shape[0] < n_frame:
        ax_seq_lld = concatenate([
            ax_seq_lld,
            zeros((n_frame - ax_seq_lld.shape[0], ax_seq_lld.shape[-1]))
        ])
    elif ax_seq_lld.shape[0] > n_frame:
        ax_seq_lld = ax_seq_lld[:n_frame]

    save((dest_path / 'seq_lld.npy').as_posix(), ax_seq_lld)


def move_final_vgg_feature(vgg_model: Model, tmp_path: Path, dest_path: Path, n_frame: int) -> None:
    # Move full audio vgg feature
    ax_vgg = transform_image(tmp_path / f'full_spec.png', vgg_model, is_dir=False)
    save((dest_path / f'full_vgg.npy').as_posix(), ax_vgg)

    # Build vgg sequence and save it to dest
    ax_vgg = transform_image(tmp_path, vgg_model, n_frame=n_frame)
    save((dest_path / f'seq_vgg.npy').as_posix(), ax_vgg)


if __name__ == '__main__':
    # Get project path
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    raw_data_path = data_path / "01 - raw"
    pth_feature_out = data_path / "02 - features"

    # The strategy here is to check the market every X minutes
    #  we have

    # Parameters
    predict_interval = 60 # 60 minutes
    backward_interval = 15840 # 24H
    windows_length = 60
    hop_size = 60


    # TODO: Features are:
    #  => deep representation of time series
    #   -> Stored as sequence (key=deep_time_seq)
    #   -> Stored as mean of 1 (key=deep_time_avg)
    #  => Wavelet transform of time series
    #   -> Stored as sequence (key=wav_seq)
    #   -> Stored as mean of 1 (key=wav_avg)
    #  => full
    #   -> deep representation of spectrogram (key=deep_spec_full)
    #   -> deep reprentation of time series (key=deep_time_full)
    #   -> Wavelet transform of time series (key=wav_full)

    # load vgg
    vgg_model = create_vgg_model()

    for filename in os.listdir(raw_data_path):

        # If already exists, continue otherwise, create feature dir
        key = str(filename.split('.')[0])
        pth_sub_out = pth_feature_out / key
        if pth_sub_out.exists():
            continue

        # Create dir and log
        pth_sub_out.mkdir()
        print(f'Computing features for {key}')

        # Create tmp dir
        tmp_dir = Temp(
            prefix=f'marktech_{key}_', suffix='_ftrs', is_dir=True,
            dir=(data_path / '06 - tmp').as_posix()
        )

        # Read audio
        ts_data = (
            TimeSeriesDriver(key=)
            .read(raw_data_path / str(filename))
        )

        # Compute features on each segment
        l_segments = [(ts_data, 'full')] + [
            (ts_data.segment(t - pd.Time , t), str(i))
            for i, t in enumerate(arange(0, int(audio.duration() * 1000), hop_size))
        ]

        with Pool(4, initializer=init_pool_worker, initargs=(tmp_dir,)) as p:
            l_res = list(p.starmap(compute_features, l_segments))

        # Save final features
        move_final_lld_feature(tmp_dir.path, pth_sub_out, r"([0-9]{1,3}_lld\.npy)", n_frame)
        move_final_vgg_feature(vgg_model, tmp_dir.path, pth_sub_out, n_frame)

        # Save final vgg features
        tmp_dir.remove()
