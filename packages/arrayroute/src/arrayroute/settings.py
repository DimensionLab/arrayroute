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
    tensorstore_cache_bytes: int = 0
    immutable: bool = False

    def __post_init__(self):
        if (isinstance(self.tensorstore_threads, bool)
                or not isinstance(self.tensorstore_threads, int)
                or not 1 <= self.tensorstore_threads <= 256):
            raise ValueError('Native threads must be an integer from 1 to 256')
        if (isinstance(self.tensorstore_cache_bytes, bool)
                or not isinstance(self.tensorstore_cache_bytes, int)
                or self.tensorstore_cache_bytes < 0):
            raise ValueError('Cache bytes must be a non-negative integer')
        self.session(1)

    @property
    def name(self) -> str:
        return (f'{self.engine}.{self.read_strategy}.g{self.gather_workers}'
                f'.t{self.tensorstore_threads}'
                + (f'.c{self.tensorstore_cache_bytes}.i{int(self.immutable)}'
                   if self.tensorstore_cache_bytes or self.immutable else ''))

    def session(self, buffer_bytes: int) -> LocalReadSession:
        return LocalReadSession(self.engine, buffer_bytes=buffer_bytes,
                                read_strategy=self.read_strategy,
                                gather_workers=self.gather_workers,
                                tensorstore_threads=self.tensorstore_threads,
                                tensorstore_cache_bytes=self.tensorstore_cache_bytes,
                                immutable=self.immutable)
