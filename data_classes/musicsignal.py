"""
MusicSignal subclass for musical signal data with MusicMetadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from data_classes.signal_base import Signal


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

    def save(self, file_path: str) -> None:
        """Save MusicSignal to NPZ using the project's schema.

        The file will contain arrays: signal, time, fs, channel_names and
        string metadata fields. The `extra` dict is saved as a pickled
        object via allow_pickle=True.
        """
        path = Path(file_path)
        np.savez(
            str(path),
            signal=self.values.astype(np.float32),
            time=self.time.astype(np.float64) if self.time is not None else np.array([], dtype=np.float64),
            fs=np.float32(self.fs),
            channel_names=np.array(self.channels, dtype=object),
            author=self.metadata.author,
            music_type=self.metadata.music_type,
            length=np.float32(self.metadata.length),
            source=self.metadata.source,
            metadata=np.array(self.metadata.extra, dtype=object),
        )

    @classmethod
    def load(cls, file_path: str) -> "MusicSignal":
        """Load MusicSignal from NPZ created by `save` or converters.

        Returns a MusicSignal instance with Signal data and Metadata.
        """
        data = np.load(file_path, allow_pickle=True)
        time = data['time'] if len(data['time']) > 0 else None
        signal = data['signal']
        fs = float(data['fs'])
        channels = list(data['channel_names'])
        metadata_obj = data['metadata'].tolist() if isinstance(data['metadata'], np.ndarray) else data['metadata']
        extra = metadata_obj if isinstance(metadata_obj, dict) else {}
        meta = MusicMetadata(
            author=str(data.get('author', 'unknown')),
            music_type=str(data.get('music_type', 'unknown')),
            length=float(data.get('length', 0.0)),
            source=str(data.get('source', 'unknown')),
            extra=extra,
        )
        return cls(values=signal, fs=fs, channels=channels, time=time, metadata=meta)

    def info(self) -> None:
        """Print information about the MusicSignal instance."""
        print("MusicSignal:")
        print(f"  Channels: {self.channels}")
        print(f"  Sampling Rate (fs): {self.fs} Hz")
        print(f"  Number of Samples: {self.values.shape[0]}")
        print(f"  Duration: {self.metadata.length:.2f} seconds")
        print(f"  Author: {self.metadata.author}")
        print(f"  Music Type: {self.metadata.music_type}")
        print(f"  Source: {self.metadata.source}")
