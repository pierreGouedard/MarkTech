# system util
from pathlib import Path
import tempfile
import shutil
import os


class Temp(object):
    def __init__(self, prefix, suffix='', is_dir=False, dir: str = None, destination: Path = None):
        """
        Temporary file
        The file is automatically deleted when this object is garbage collected.
        If possible, delete the file manually by calling remove().
        Example usage:
            ```
            mytmp = TempFile(driver)
            with driver.write(mytmp.path) as writer:
                writer.write('hello world')
            # Do whatever your want with the file (like uploading it somewhere)
            mytmp.remove()
            ```
        :param BaseDriver driver:
        :param str prefix:
        :param str suffix:
        :param str dir: Where to create the temporary file
        :param str destination: The optional destination of the content of this folder
        """
        self.destination = destination
        self.is_dir = is_dir
        if is_dir:
            self.path = Path(os.path.abspath(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)))
        else:
            self.path = Path(os.path.abspath(tempfile.mktemp(suffix=suffix, prefix=prefix, dir=dir)))

    def move_to_destination(self):
        """
        Move the temporary file to its destination.
        Also works when `self.path` is a directory (hard sync).
        If the destination already exist, it will be deleted before the move.
        """
        if self.path is None:
            raise ValueError('Temporary path has already been moved to destination "{}"'.format(self.destination))

        if self.destination is None:
            raise ValueError('Cannot move to None destnation. Please set a destination ')

        if self.destination.exists:
            shutil.rmtree(self.destination.as_posix())

        # Move
        os.rename(self.path, self.destination)
        self.remove()

    def remove(self):
        """
        Remove the temporary file/directory
        Once called, self.path is set to None
        """
        if self.path is not None and self.path.exists():
            if self.is_dir:
                shutil.rmtree(self.path, ignore_errors=True)
            else:
                os.remove(self.path)

        self.path = None

    def __del__(self):
        self.remove()
