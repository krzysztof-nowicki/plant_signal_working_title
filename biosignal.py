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
from matplotlib import pyplot as plt


@dataclass
class Signal:
    """Raw multichannel signal with timing information.

    Attributes:
        values: numpy array of shape (n_samples, n_channels)
        fs: sampling frequency in Hz
        channels: list of channel names
        time: optional time axis (numpy array)
    """

    values: np.ndarray
    fs: float
    channels: List[str]
    time: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.time is None and self.values is not None and self.fs:
            n_samples = int(self.values.shape[0])
            self.time = np.linspace(0, n_samples / float(self.fs), n_samples, dtype=np.float64)


@dataclass
class Metadata:
    """Recording metadata.

    Keep this small and focused; arbitrary key/value pairs can be stored
    in the ``extra`` dict.
    """

    organism: str = "unknown"
    species: str = "unknown"
    recording_type: str = "unknown"
    units: str = "unknown"
    source: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


class BioSignal:
    """High-level BioSignal composed from Signal + Metadata.

    The class intentionally has a narrow public surface: a single
    attribute for the signal and one for metadata. Use ``from_parts`` to
    construct from primitive values (keeps converters concise without a
    fat constructor).
    """

    def __init__(self, signal: Signal, metadata: Metadata):
        self.signal = signal
        self.metadata = metadata

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
        sig = Signal(values=np.asarray(values, dtype=np.float32), fs=fs, channels=list(channels), time=time)
        meta = Metadata(
            organism=(organism or "unknown"),
            species=(species or "unknown"),
            recording_type=(recording_type or "unknown"),
            units=(units or "unknown"),
            source=(source or "unknown"),
            extra=(extra or {}),
        )
        return cls(sig, meta)

    def save(self, file_path: str) -> None:
        """Save BioSignal to NPZ using the project's schema.

        The file will contain arrays: signal, time, fs, channel_names and
        string metadata fields. The `extra` dict is saved as a pickled
        object via allow_pickle=True.
        """
        path = Path(file_path)
        np.savez(
            str(path),
            signal=self.signal.values.astype(np.float32),
            time=self.signal.time.astype(np.float64) if self.signal.time is not None else np.array([], dtype=np.float64),
            fs=np.float32(self.signal.fs),
            channel_names=np.array(self.signal.channels, dtype=object),
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

        Returns a BioSignal instance with composed Signal and Metadata.
        """
        data = np.load(file_path, allow_pickle=True)
        time = data['time'] if len(data['time']) > 0 else None
        signal = data['signal']
        fs = float(data['fs'])
        channels = list(data['channel_names'])
        metadata_obj = data['metadata'].tolist() if isinstance(data['metadata'], np.ndarray) else data['metadata']
        extra = metadata_obj if isinstance(metadata_obj, dict) else {}
        meta = Metadata(
            organism=str(data.get('organism', 'unknown')),
            species=str(data.get('species', 'unknown')),
            recording_type=str(data.get('recording_type', 'unknown')),
            units=str(data.get('units', 'unknown')),
            source=str(data.get('source', 'unknown')),
            extra=extra,
        )
        sig = Signal(values=signal, fs=fs, channels=channels, time=time)
        return cls(sig, meta)

    # Backward-compatible aliases
    @property
    def fs(self) -> float:
        return float(self.signal.fs)

    def plot_original(self) -> None:
        """Plot the raw recorded signal without normalization."""
        if self.signal.values is None:
            print("No signal data to plot.")
            return

        plt.figure(figsize=(10, 6))
        for i, channel in enumerate(self.signal.channels):
            plt.subplot(len(self.signal.channels), 1, i + 1)
            plt.plot(self.signal.time, self.signal.values[:, i])
            plt.xlabel("Time (s)")
            plt.ylabel(f"Channel {channel}")
            plt.title(f"Signal Plot - {channel}")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot(self, duration: Optional[float] = 60.0, max_signal: Optional[float] = 100.0, max_samples: Optional[int] = 100) -> None:
        """Plot with optional resampling and normalization.

        The method operates on a copy of the data so it does not mutate
        the stored signal.
        """
        if self.signal.values is None:
            print("No signal data to plot.")
            return

        signal = self.signal.values.copy()
        time = self.signal.time.copy() if self.signal.time is not None else None

        if max_samples is not None:
            signal = self._resample_signal(signal, max_samples)
            if time is not None:
                time = np.linspace(0, float(duration), max_samples, dtype=np.float64)
        else:
            if time is not None:
                current_duration = float(time[-1])
                if current_duration != float(duration):
                    time = np.linspace(0, float(duration), len(time), dtype=np.float64)

        if max_signal is not None:
            signal = self._normalize_signal(signal, max_signal)

        plt.figure(figsize=(10, 6))
        for i, channel in enumerate(self.signal.channels):
            plt.subplot(len(self.signal.channels), 1, i + 1)
            plt.plot(time, signal[:, i])
            plt.xlabel("Time (s)")
            plt.ylabel(f"Channel {channel}")
            plt.title(f"Signal Plot - {channel}")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _resample_signal(signal: np.ndarray, target_samples: int) -> np.ndarray:
        n_channels = signal.shape[1]
        resampled = np.zeros((target_samples, n_channels), dtype=np.float32)
        old_indices = np.linspace(0, signal.shape[0] - 1, signal.shape[0])
        new_indices = np.linspace(0, signal.shape[0] - 1, target_samples)
        for i in range(n_channels):
            resampled[:, i] = np.interp(new_indices, old_indices, signal[:, i])
        return resampled

    @staticmethod
    def _normalize_signal(signal: np.ndarray, max_value: float) -> np.ndarray:
        current_max = np.max(np.abs(signal))
        if current_max > 0:
            signal = signal * (float(max_value) / float(current_max))
        return signal

    def info(self) -> None:
        print("BioSignal Information:")
        print(f"  Organism: {self.metadata.organism}")
        print(f"  Species: {self.metadata.species}")
        print(f"  Recording Type: {self.metadata.recording_type}")
        print(f"  Units: {self.metadata.units}")
        print(f"  Source: {self.metadata.source}")
        print(f"  Sampling Rate (fs): {self.signal.fs} Hz")
        print(f"  Number of Channels: {len(self.signal.channels)}")
        print(f"  Channel Names: {', '.join(self.signal.channels)}")
        if self.signal.values is not None:
            print(f"  Signal Shape: {self.signal.values.shape}")
            print(f"  Time Axis Length: {len(self.signal.time) if self.signal.time is not None else 'N/A'}")
        else:
            print("  Signal data is not available.")
