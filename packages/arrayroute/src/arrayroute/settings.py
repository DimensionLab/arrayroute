# SPDX-License-Identifier: Apache-2.0
"""Validated local reader settings; plain data, never executable persisted factories."""
from dataclasses import dataclass

from .session import LocalReadSession


@dataclass(frozen=True)
class ReadSettings:
    engine: str = 'zarr-python'
    read_strategy: str = 'native'
    gather_workers: int = 4
    tensorstore_threads: int = 1

    def __post_init__(self):
        if (isinstance(self.tensorstore_threads, bool)
                or not isinstance(self.tensorstore_threads, int)
                or not 1 <= self.tensorstore_threads <= 256):
            raise ValueError('Native threads must be an integer from 1 to 256')
        self.session(1)

    @property
    def name(self) -> str:
        return (f'{self.engine}.{self.read_strategy}.g{self.gather_workers}'
                f'.t{self.tensorstore_threads}')

    def session(self, buffer_bytes: int) -> LocalReadSession:
        return LocalReadSession(self.engine, buffer_bytes=buffer_bytes,
                                read_strategy=self.read_strategy,
                                gather_workers=self.gather_workers,
                                tensorstore_threads=self.tensorstore_threads)
