# system util
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import pytz


class TimeSeriesDriver:

    def __init__(self, time_col: str, value_col: str, sr: float, data: pd.Series = None):

        # Mandatory attribute
        self.time_col, self.value_col = time_col, value_col
        self.sr = sr

        # Optional
        self.data = data

    def duration(self) -> float:
        pass

    def read(self, path: Path, utc_convert: bool = True, dayfirst: bool = True):
        # load time series
        data = pd.read_csv(path.as_posix())
        self.data = (
            data
            .assign(**{self.time_col: pd.to_datetime(data[self.time_col], dayfirst=dayfirst)})
            .loc[:, [self.time_col, self.value_col]]
            .set_index(self.time_col, drop=True)
            .loc[:, self.value_col]
        ).sort_index()
        if utc_convert:
            self.data.index = self.data.index.tz_convert(pytz.utc)

        return self

    def write(self, path: Path):
        self.data.to_csv(path.as_posix())
        return self

    def segment_time(self, dt_start: pd.Timestamp, dt_stop: pd.Timestamp, inplace=False):
        # Load segment and cut it at specified bounds
        segment = self.data.loc[(self.data.index >= dt_start) & (self.data.index <= dt_stop)]

        if inplace:
            self.data = segment
            return self

        return TimeSeriesDriver(self.time_col, self.value_col, self.sr, segment)

    def segment_ind(self, ind_start: int, ind_stop: int, inplace=False):
        # Load segment and cut it at specified bounds
        segment = self.data.iloc[ind_start:ind_stop]

        if inplace:
            self.data = segment
            return self

        return TimeSeriesDriver(self.time_col, self.value_col, self.sr, segment)

    def plot(self):
        self.data.plot()
        plt.show()
