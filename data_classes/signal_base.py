"""
Core Signal dataclass and small specialized subclasses.

This module contains the base Signal class (raw values + timing + plotting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class Signal:
    """Raw multichannel signal with timing information and plotting helpers.

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
            # time from 0 to duration (inclusive of last sample)
            self.time = np.linspace(0, n_samples / float(self.fs), n_samples, dtype=np.float64)

    def plot_original(self) -> None:
        """Plot the raw recorded signal without normalization."""
        if self.values is None:
            print("No signal data to plot.")
            return

        plt.figure(figsize=(10, 6))
        for i, channel in enumerate(self.channels):
            plt.subplot(len(self.channels), 1, i + 1)
            plt.plot(self.time, self.values[:, i])
            plt.xlabel("Time (s)")
            plt.ylabel(f"Channel {channel}")
            plt.title(f"Signal Plot - {channel}")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot(self, duration: Optional[float] = 60.0, max_signal: Optional[float] = 100.0,
             max_samples: Optional[int] = 100) -> None:
        """Plot with optional resampling and normalization.

        'duration' controls the x-axis span used when resampling/creating a synthetic time axis.
        'max_signal' scales the signal to the desired peak value.
        'max_samples' resamples the signal to at most that number of samples for plotting.
        """
        if self.values is None:
            print("No signal data to plot.")
            return

        signal = self.values.copy()
        time = self.time.copy() if self.time is not None else None

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
        for i, channel in enumerate(self.channels):
            plt.subplot(len(self.channels), 1, i + 1)
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
