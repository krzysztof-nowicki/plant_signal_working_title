"""Structured BioSignal container with smaller, focused data classes.

This module defines a Signal dataclass (raw data + timing), a Metadata
dataclass (organism/species/source/etc.) and a BioSignal wrapper that
composes them. Converters should use BioSignal.from_parts(...) to build
instances; BioSignal.save/load provide NPZ persistence compatible with
previous format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from data_classes.signal_base import Signal


@dataclass
class BioMetadata:
    """Recording metadata.

    Keep this small and focused; arbitrary key/value pairs can be stored
    in the 'extra' dict.
    """

    organism: str = "unknown"
    species: str = "unknown"
    recording_type: str = "unknown"
    units: str = "unknown"
    source: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


class BioSignal(Signal):
    """High-level BioSignal with Signal data + BioMetadata.

    Inherits plotting and signal methods from Signal.
    Use 'from_parts' to construct from primitive values.
    """

    metadata: BioMetadata = None

    def __init__(self, values: np.ndarray, fs: float, channels: List[str],
                 time: Optional[np.ndarray] = None, metadata: Optional[BioMetadata] = None):
        super().__init__(values=values, fs=fs, channels=channels, time=time)
        self.metadata = metadata or BioMetadata()

    @classmethod
    def from_parts(
            cls,
            values: np.ndarray,
            fs: float,
            channels: List[str],
            time: Optional[np.ndarray] = None,
            *,
            organism: Optional[str] = None,
            species: Optional[str] = None,
            recording_type: Optional[str] = None,
            units: Optional[str] = None,
            source: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
    ) -> "BioSignal":
        """Convenience constructor used by converters.

        All metadata keyword arguments are optional and default to "unknown".
        """
        meta = BioMetadata(
            organism=(organism or "unknown"),
            species=(species or "unknown"),
            recording_type=(recording_type or "unknown"),
            units=(units or "unknown"),
            source=(source or "unknown"),
            extra=(extra or {}),
        )
        return cls(
            values=np.asarray(values, dtype=np.float32),
            fs=fs,
            channels=list(channels),
            time=time,
            metadata=meta)

    def save(self, file_path: str) -> None:
        """Save BioSignal to NPZ using the project's schema.

        The file will contain arrays: signal, time, fs, channel_names and
        string metadata fields. The `extra` dict is saved as a pickled
        object via allow_pickle=True.
        """
        path = Path(file_path)
        np.savez(
            str(path),
            signal=self.values.astype(np.float32),
            time=self.time.astype(np.float64) if self.time is not None else np.array([],
                                                                                     dtype=np.float64),
            fs=np.float32(self.fs),
            channel_names=np.array(self.channels, dtype=object),
            organism=self.metadata.organism,
            species=self.metadata.species,
            recording_type=self.metadata.recording_type,
            units=self.metadata.units,
            source=self.metadata.source,
            metadata=np.array(self.metadata.extra, dtype=object),
        )

    @classmethod
    def load(cls, file_path: str) -> "BioSignal":
        """Load BioSignal from NPZ created by `save` or converters.

        Returns a BioSignal instance with Signal data and Metadata.
        """
        data = np.load(file_path, allow_pickle=True)
        time = data['time'] if len(data['time']) > 0 else None
        signal = data['signal']
        fs = float(data['fs'])
        channels = list(data['channel_names'])
        metadata_obj = data['metadata'].tolist() if isinstance(data['metadata'], np.ndarray) else data['metadata']
        extra = metadata_obj if isinstance(metadata_obj, dict) else {}
        meta = BioMetadata(
            organism=str(data.get('organism', 'unknown')),
            species=str(data.get('species', 'unknown')),
            recording_type=str(data.get('recording_type', 'unknown')),
            units=str(data.get('units', 'unknown')),
            source=str(data.get('source', 'unknown')),
            extra=extra,
        )
        return cls(values=signal, fs=fs, channels=channels, time=time, metadata=meta)

    def info(self) -> None:
        """Print information about all BioSignal attributes."""
        print("BioSignal Information:")
        print(f"  Organism: {self.metadata.organism}")
        print(f"  Species: {self.metadata.species}")
        print(f"  Recording Type: {self.metadata.recording_type}")
        print(f"  Units: {self.metadata.units}")
        print(f"  Source: {self.metadata.source}")
        print(f"  Sampling Rate (fs): {self.fs} Hz")
        print(f"  Number of Channels: {len(self.channels)}")
        print(f"  Channel Names: {', '.join(self.channels)}")
        if self.values is not None:
            print(f"  Signal Shape: {self.values.shape}")
            print(f"  Time Axis Length: {len(self.time) if self.time is not None else 'N/A'}")
        else:
            print("  Signal data is not available.")
