import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict


@dataclass
class Timer:
    durations: Dict[str, float]

    def __init__(self) -> None:
        self.durations = {}

    @contextmanager
    def record(self, key: str):
        start = time.perf_counter()
        yield
        self.durations[key] = time.perf_counter() - start

    def add(self, key: str, value: float) -> None:
        self.durations[key] = value
