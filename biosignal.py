"""Core BioSignal container and helpers.

This module provides the BioSignal class used across the project to
represent multichannel time series along with simple I/O and plotting
utilities.

The module name is intentionally kept PascalCase for historical
compatibility with the rest of the project; pylint's "invalid-name"
warning is disabled for the module.
"""

# pylint: disable=invalid-name
import numpy as np
from matplotlib import pyplot as plt


class BioSignal:
    """
    Container for multichannel biosignal recordings.
    """
    signal = None
    time = None
    fs = None
    channels = None
    organism = None
    species = None
    recording_type = None
    units = None
    source = None
    metadata = None

    def __init__(self, signal, fs, channels, time=None, organism=None, species=None,
                 recording_type=None, units=None, source=None, metadata=None):
        self.signal = signal
        self.fs = fs
        self.channels = channels
        self.time = time if time is not None else self._generate_time_axis()
        self.organism = organism or "unknown"
        self.species = species or "unknown"
        self.recording_type = recording_type or "unknown"
        self.units = units or "unknown"
        self.source = source or "unknown"
        self.metadata = metadata or {}

    def _generate_time_axis(self):
        """Generate time axis based on signal length and sampling rate."""
        if self.signal is None or self.fs is None:
            return None
        n_samples = self.signal.shape[0]
        return np.linspace(0, n_samples / self.fs, n_samples, dtype=np.float64)

    # Backward compatibility properties
    @property
    def fd(self):
        """Backward-compatible alias for sampling frequency (fs)."""
        return self.fs

    @property
    def sample_rate(self):
        """Alternate alias for sampling frequency (fs)."""
        return self.fs

    def save(self, file_path):
        """
        Save BioSignal to NPZ file with rich metadata schema.
        
        Args:
            file_path: Path to save NPZ file
        """
        np.savez(
            file_path,
            signal=self.signal.astype(np.float32),
            time=self.time.astype(np.float64) if self.time is not None else np.array([]),
            fs=np.float32(self.fs),
            channel_names=np.array(self.channels, dtype=object),
            organism=self.organism,
            species=self.species,
            recording_type=self.recording_type,
            units=self.units,
            source=self.source,
            metadata=np.array(self.metadata, dtype=object)
        )

    @staticmethod
    def load(file_path):
        """
        Load BioSignal from NPZ file.
        
        Args:
            file_path: Path to NPZ file
            
        Returns:
            BioSignal object
        """
        data = np.load(file_path, allow_pickle=True)

        time = data['time'] if len(data['time']) > 0 else None

        return BioSignal(
            signal=data['signal'],
            fs=float(data['fs']),
            channels=list(data['channel_names']),
            time=time,
            organism=str(data['organism']),
            species=str(data['species']),
            recording_type=str(data['recording_type']),
            units=str(data['units']),
            source=str(data['source']),
            metadata=data['metadata'].item()
        )

    def plot_original(self):
        """
        Plot the signal without any changes.
        """
        if self.signal is None:
            print("No signal data to plot.")
            return

        plt.figure(figsize=(10, 6))
        for i, channel in enumerate(self.channels):
            plt.subplot(len(self.channels), 1, i + 1)
            plt.plot(self.time, self.signal[:, i])
            plt.xlabel("Time (s)")
            plt.ylabel(f"Channel {channel}")
            plt.title(f"Signal Plot - {channel}")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot(self, duration=60, max_signal=100, max_samples=100):
        """
        Plot the signal with optional normalization.
        
        Args:
            duration: Target duration in seconds (default 60). Signals shorter than this
                     will be extended, longer signals will be compressed.
            max_signal: Maximum signal amplitude for normalization. If provided, the signal
                       will be scaled so the largest absolute value matches this value.
            max_samples: Maximum number of samples to resample to. If provided, the signal
                        will be resampled to have exactly this many samples.
        """
        if self.signal is None:
            print("No signal data to plot.")
            return

        signal = self.signal.copy()
        time = self.time.copy() if self.time is not None else None

        if max_samples is not None:
            signal = self._resample_signal(signal, max_samples)
            if time is not None:
                time = np.linspace(0, duration, max_samples, dtype=np.float64)
        else:
            if time is not None:
                current_duration = time[-1]
                if current_duration != duration:
                    time = np.linspace(0, duration, len(time), dtype=np.float64)

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

    def _resample_signal(self, signal, target_samples):
        """
        Resample signal to have exactly target_samples samples.
        Uses linear interpolation.
        
        Args:
            signal: Signal array of shape (n_samples, n_channels)
            target_samples: Target number of samples
            
        Returns:
            Resampled signal array
        """
        n_channels = signal.shape[1]
        resampled = np.zeros((target_samples, n_channels))

        old_indices = np.linspace(0, signal.shape[0] - 1, signal.shape[0])
        new_indices = np.linspace(0, signal.shape[0] - 1, target_samples)

        for i in range(n_channels):
            resampled[:, i] = np.interp(new_indices, old_indices, signal[:, i])

        return resampled

    def _normalize_signal(self, signal, max_value):
        """
        Normalize signal so that the largest absolute value equals max_value.
        Preserves the sign and relative magnitudes of peaks.
        
        Args:
            signal: Signal array of shape (n_samples, n_channels)
            max_value: Target maximum absolute value
            
        Returns:
            Normalized signal array
        """
        current_max = np.max(np.abs(signal))

        if current_max > 0:
            signal = signal * (max_value / current_max)

        return signal

    def info(self):
        """
        Print detailed information about the BioSignal object.
        """
        print("BioSignal Information:")
        print(f"  Organism: {self.organism}")
        print(f"  Species: {self.species}")
        print(f"  Recording Type: {self.recording_type}")
        print(f"  Units: {self.units}")
        print(f"  Source: {self.source}")
        print(f"  Sampling Rate (fs): {self.fs} Hz")
        print(f"  Number of Channels: {len(self.channels)}")
        print(f"  Channel Names: {', '.join(self.channels)}")
        if self.signal is not None:
            print(f"  Signal Shape: {self.signal.shape}")
            print(f"  Time Axis Length: {len(self.time) if self.time is not None else 'N/A'}")
        else:
            print("  Signal data is not available.")
