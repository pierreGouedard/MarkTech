from typing import List, Tuple
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField
from kymatio.numpy import Scattering1D


def compute_gramian_angular_field(data: np.ndarray, image_size: int):
    gaf = GramianAngularField(image_size=min(image_size, len(data)))
    return gaf.fit_transform(data[np.newaxis, :])[0]


def compute_markov_transition_field(data: np.ndarray, image_size: int):
    mtf = MarkovTransitionField(image_size=min(image_size, len(data)))
    return mtf.fit_transform(data[np.newaxis, :])[0]


def compute_scatter_feature(data: np.ndarray, J: int, Q: int, show: bool = False):

    # Compute scatter transform
    scattering = Scattering1D(J, data.shape[-1], Q)
    Sx = scattering((data - data.mean()) / data.std())

    # Get meta
    meta = scattering.meta()
    l_orders = [
        np.where(meta['order'] == 0), np.where(meta['order'] == 1), np.where(meta['order'] == 2)
    ]

    # Show scatter transform if asked
    if show:
        for i, order in enumerate(l_orders):
            plt.plot(Sx[order] if i > 0 else Sx[order][0], title=f"order {i} of scatter transform")

    return [Sx[order] if i > 0 else Sx[order][0] for i, order in enumerate(l_orders)]


def save_as_image(output_path: Path, im: np.ndarray) -> None:

    # Build image and save it
    fig, ax = plt.subplots()
    ax.set_axis_off()
    plt.imshow(im)
    plt.savefig(output_path.as_posix(), bbox_inches='tight', transparent=True, pad_inches=0.0)

    # Clear figure
    plt.cla()
    plt.close(fig)
    plt.clf()
