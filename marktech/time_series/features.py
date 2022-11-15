from typing import List, Tuple
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
import opensmile

# Local import
from marktech.audio.driver import AudioDriver


def compute_opensmile_features(l_files: List[str], n_worker: int = 1) -> pd.DataFrame:
    egemaps_smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=n_worker,
    )

    df_egemaps = egemaps_smile.process_files(l_files)

    return df_egemaps


def compute_mfcc(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    return mfcc, mfcc_delta, mfcc_delta2


def compute_llds(audio: AudioDriver, mfcc_agg: str = 'mean', num_worker: int = 1) -> np.ndarray:
    # Compute mfcc features
    mfcc, mfcc_delta, mfcc_delta2 = compute_mfcc(audio.data, audio.sr)

    if mfcc_agg == 'mean':
        mfcc_features = np.hstack([mfcc.mean(axis=1), mfcc_delta.mean(axis=1), mfcc_delta2.mean(axis=1)])
    elif mfcc_agg == 'min':
        mfcc_features = np.hstack([mfcc.min(axis=1), mfcc_delta.min(axis=1), mfcc_delta2.min(axis=1)])
    elif mfcc_agg == 'max':
        mfcc_features = np.hstack([mfcc.max(axis=1), mfcc_delta.max(axis=1), mfcc_delta2.max(axis=1)])
    else:
        raise ValueError('{mfcc_agg} mfcc agg undefined')

    # Compute eGeMAPS features
    egemaps = compute_opensmile_features([audio.temp_file.path], num_worker).values[0]

    return np.hstack([mfcc_features, egemaps])


def compute_stft_spectrogram(y: np.ndarray, sr: int, window_size: int = 2048, show: bool = False) -> np.ndarray:
    window = np.hanning(window_size)
    stft_spec = librosa.core.spectrum.stft(y)
    stft_spec = 2 * np.abs(stft_spec) / np.sum(window)

    # Show spectrogram
    if show:
        show_spectrogram(stft_spec, sr)
        plt.cla()
        plt.clf()

    return stft_spec


def compute_mel_spectrogram(
        y: np.ndarray, sr: int, show: bool = False, n_mels: int = 128
) -> np.ndarray:

    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    if show:
        show_spectrogram(melspec, sr)
        plt.cla()
        plt.clf()

    return melspec


def show_spectrogram(spec: np.ndarray, sr: int) -> None:
    fig, ax = plt.subplots()
    spec_db = librosa.power_to_db(spec, ref=np.max)
    _ = librosa.display.specshow(spec_db, x_axis='time', y_axis='log', sr=sr, fmax=8000, ax=ax)
    ax.set(title='spectrogram')
    plt.show()


def save_spectrogram_as_image(output_path: Path, spec: np.ndarray, sr: int) -> None:
    fig, ax = plt.subplots()
    ax.set_axis_off()
    spec_db = librosa.power_to_db(spec, ref=np.max)
    _ = librosa.display.specshow(spec_db, x_axis='time', y_axis='log', sr=sr, fmax=8000, ax=ax)
    plt.savefig(output_path.as_posix(), bbox_inches='tight', transparent=True, pad_inches=0.0)

    plt.cla()
    plt.close(fig)
    plt.clf()


