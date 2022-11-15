# system util
from pathlib import Path

# Audio processing lib
import pandas as pd
import numpy as np

# Local import
from marktech.utils.temp import Temp


class TimeSeriesDriver:

    def __init__(self, time_col: str, value_col: str, data: pd.Series = None):

        # Mandatory attribute
        self.time_col, self.value_col = time_col, value_col

        # Optional
        self.data = data

    def duration(self) -> float:
        pass

    def read(self, path: Path):
        # load time series
        data = pd.read_csv(path.as_posix())

        self.data = (
            data
            .assign({self.time_col: pd.to_datetime(data[self.time_col])})
            .loc[:, [self.time_col, self.value_col]]
            .set_index(self.time_col, drop=True)
            .loc[:, self.value_col]
        )

        return self

    def write(self, path: Path):
        if new_sr is not None:
            self.resample(new_sr)

        self.data.to_csv(path.as_posix())
        return self

    def segment(self, dt_start: pd.Timestamp, dt_stop: pd.Timestamp, inplace=False):
        # Load segment and cut it at specified bounds
        segment = self.data.loc[(self.data >= dt_start) && (self.data < dt_stop)]

        if inplace:
            self.data = segment
            return self

        return TimeSeriesDriver(self.time_col, self.value_col, segment)

    def plot(self):
        import IPython
        IPython.embed()
        pass
