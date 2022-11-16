from dataclasses import dataclass


@dataclass
class FeatureMeta:
    key: str
    id: int
    date_start: str
    date_end: str
    is_test: bool