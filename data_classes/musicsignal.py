"""
MusicSignal subclass that adds a baseline and region/limit parameters
commonly useful when rendering musical signals.
"""
from dataclasses import dataclass, field
from typing import Dict, Any

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


@dataclass
class MusicSignal(Signal):
    """Specialized Signal for musical data.

    Adds a baseline value (DC offset) and convenience plotting parameters
    to limit shown region (start/end times). The plotting method keeps the
    same signature as Signal.plot with extra optional parameters for
    region limiting.
    """

    def __init__(self, signal: Signal, metadata: MusicMetadata):
        self.signal = signal
        self.metadata = metadata