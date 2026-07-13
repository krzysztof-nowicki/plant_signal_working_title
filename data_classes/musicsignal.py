"""
MusicSignal subclass for musical signal data with MusicMetadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
from data_classes.signal import Signal


@dataclass
class MusicMetadata:
    """Recording metadata.

    Keep this small and focused; arbitrary key/value pairs can be stored
    in the 'extra' dict.
    """

    author: str = "unknown"
    music_type: str = "unknown"
    length: float = 0.0
    source: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


class MusicSignal(Signal):
    """Specialized Signal for musical data.

    Inherits plotting and signal methods from Signal.
    Adds MusicMetadata for music-specific information.
    """

    metadata: MusicMetadata = None

    def __init__(self, values: np.ndarray, fs: float, channels: List[str],
                 time: Optional[np.ndarray] = None, metadata: Optional[MusicMetadata] = None):
        super().__init__(values=values, fs=fs, channels=channels, time=time)
        self.metadata = metadata or MusicMetadata()
