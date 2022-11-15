from typing import List
from pathlib import Path
import tarfile
import wget
import os


def download_data(url: str, pth_out: Path, max_retry: int = 5):
    n_retry, success = 0, False
    while n_retry < max_retry and not success:
        try:
            wget.download(url, out=pth_out.as_posix())
            success = True
        except:
            success = False
            n_retry += 1
            print(f'issue loading data, retrying...', flush=True)


def extract_tar(l_files: List[str], pth_tar: Path, pth_out: Path):
    t = tarfile.open(pth_tar.as_posix(), 'r')
    for member in t.getmembers():
        if Path(member.name).name in l_files:
            t.extract(member, pth_tar.parent)

            # Create output dir if necessary
            pth_dir = (pth_out / member.name).parent
            if not pth_dir.exists():
                pth_dir.mkdir()

            # Move member to destination
            os.rename(pth_tar.parent / member.name, pth_out / member.name)


