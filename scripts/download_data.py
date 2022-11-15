""" Download raw data from online edaic dataset.

The script download only the audio data and the transcript.
"""
# Global import
from typing import List
from pathlib import Path
import sys

project_path = Path(__file__).parent.parent
sys.path.append(project_path.as_posix())

# Local import
from marktech.utils.get_data import extract_tar, download_data
from marktech.utils.temp import Temp


if __name__ == '__main__':
    # Get project path and load (if any) local env
    project_path = Path(__file__).parent.parent
    data_path = project_path / "data"
    raw_data_path = data_path / "01 - raw"
    pth_audio_test = data_path / "06 - tests" / 'test.wav'
    pth_img_out = data_path / "06 - tests" / 'test.png'

    # Parameters
    n_start, n_end = 300, 718
    l_files: List[str] = ['{}_AUDIO.wav', '{}_Transcript.csv']
    url = 'https://dcapswoz.ict.usc.edu/wwwedaic/data/{}_P.tar'

    # Get data
    for i in range(n_start, n_end + 1):
        temp_dir = Temp(prefix='marktech_', suffix='_dwnld', is_dir=True, dir=(data_path / '06 - tmp').as_posix())

        if (raw_data_path / f'{i}_P').exists():
            continue

        # Download data
        print(f'downloading data {i}')
        download_data(url.format(int(i)), temp_dir.path)

        # Extract files of interest
        if (temp_dir.path / f"{i}_P.tar").exists():
            extract_tar(
                list(map(lambda x: x.format(i), l_files)), temp_dir.path / f"{i}_P.tar",
                raw_data_path
            )
        else:
            print(f'Problem downloading file {i}')

        # remove temp dir
        temp_dir.remove()




