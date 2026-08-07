"""
Dataset base
"""
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from baseline.config import UPSCALE, INPUT_LENGTH, OUTPUT_LENGTH, CHANNEL, DOWNSAMPLING_METHOD, DATA_DIR

try:
    from scipy.signal import decimate, resample_poly
except ImportError:  # pragma: no cover - exercised when scipy is unavailable
    def _resample_along_axis(signal: np.ndarray, target_length: int, axis: int = 1) -> np.ndarray:
        if axis != 1:
            raise NotImplementedError("Only axis=1 is supported in the fallback resampler")
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        if signal.shape[axis] == target_length:
            return signal
        source = np.arange(signal.shape[axis], dtype=np.float64)
        target = np.linspace(0, signal.shape[axis] - 1, target_length, dtype=np.float64)
        resampled = np.empty((signal.shape[0], target_length), dtype=np.float32)
        for channel_idx in range(signal.shape[0]):
            resampled[channel_idx] = np.interp(target, source, signal[channel_idx].astype(np.float64))
        return resampled

    def decimate(signal: np.ndarray, q: int, axis: int = 1, zero_phase: bool = True) -> np.ndarray:
        if q <= 1:
            return signal
        return _resample_along_axis(signal, max(1, signal.shape[axis] // q), axis=axis)

    def resample_poly(signal: np.ndarray, up: int = 1, down: int = 1, axis: int = 1) -> np.ndarray:
        if up <= 0 or down <= 0:
            raise ValueError("up and down must be positive")
        if up == 1 and down > 1:
            return _resample_along_axis(signal, max(1, signal.shape[axis] // down), axis=axis)
        if up > 1 and down == 1:
            return _resample_along_axis(signal, signal.shape[axis] * up, axis=axis)
        return signal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_classes.signal_base import Signal


def _resolve_data_dir(data_dir: str) -> str:
    path = os.path.expanduser(data_dir)
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    else:
        path = os.path.abspath(path)
    return path


def _collect_npz_files(data_dir: str) -> list[str]:
    path = _resolve_data_dir(data_dir)
    files = []
    for root, _, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.npz'):
                files.append(os.path.join(root, fname))
    files.sort()
    return files


def _load_signal_from_npz(path: str) -> Signal:
    with np.load(path, allow_pickle=True) as data:
        values = None
        for key in ('values', 'signal', 'data'):
            if key in data:
                values = data[key]
                break
        if values is None:
            raise KeyError(f"No supported signal array found in {path}")

        values = np.array(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]

        fs = float(data.get('fs', 0.0))
        channels_data = data.get('channels')
        if channels_data is None:
            channels_data = data.get('channel_names')
        if channels_data is None:
            channels = [f"channel_{idx}" for idx in range(values.shape[1])]
        else:
            channels = [str(channel) for channel in np.atleast_1d(channels_data).tolist()]
            if len(channels) != values.shape[1]:
                channels = [f"channel_{idx}" for idx in range(values.shape[1])]

    return Signal(values=values, fs=fs, channels=channels)

class SignalDataset(Dataset):
    """
    Signal dataset, made using .npz files
    """
    def __init__(self, data_dir: str = DATA_DIR):
        super().__init__()
        self.files = _collect_npz_files(data_dir)
        if not self.files:
            raise RuntimeError(f"No .npz files found in directory tree: {_resolve_data_dir(data_dir)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        signal = _load_signal_from_npz(path)
        values = signal.values
        if values.ndim == 1:
            values = values[:, None]  # (n_samples, 1)
        values = values.T  # (C, T)
        num_channels, total_length = values.shape

        out_len = OUTPUT_LENGTH if OUTPUT_LENGTH is not None else UPSCALE * INPUT_LENGTH
        if total_length < out_len:
            raise ValueError(f"Signal too short: {total_length} < {out_len}")
        # Random crop of HR fragment
        start = random.randint(0, total_length - out_len)
        hr = values[:, start:start+out_len]  # (C, out_len)

        if CHANNEL is not None and CHANNEL < num_channels:
            hr = hr[[CHANNEL], :]  # (1, out_len)

        if DOWNSAMPLING_METHOD == 'decimate':
            lr = decimate(hr, UPSCALE, axis=1, zero_phase=True)
        else:
            lr = resample_poly(hr, up=1, down=UPSCALE, axis=1)
        lr = lr[:, :INPUT_LENGTH]

        lr_tensor = torch.from_numpy(lr.astype(np.float32))
        hr_tensor = torch.from_numpy(hr.astype(np.float32))
        return lr_tensor, hr_tensor


class SignalDataclassDataset(Dataset):
    """Dataset that loads each sample through the shared Signal dataclass."""

    def __init__(self, data_dir: str = DATA_DIR):
        super().__init__()
        self.files = _collect_npz_files(data_dir)
        if not self.files:
            raise RuntimeError(f"No .npz files found in directory tree: {_resolve_data_dir(data_dir)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        signal = _load_signal_from_npz(self.files[idx])
        values = signal.values
        if values.ndim == 1:
            values = values[:, None]

        num_channels = values.shape[1]
        total_length = values.shape[0]

        out_len = OUTPUT_LENGTH if OUTPUT_LENGTH is not None else UPSCALE * INPUT_LENGTH
        if total_length < out_len:
            raise ValueError(f"Signal too short: {total_length} < {out_len}")

        start = random.randint(0, total_length - out_len)
        hr = values[start:start + out_len]

        if CHANNEL is not None and CHANNEL < num_channels:
            hr = hr[:, [CHANNEL]]

        lr = Signal._resample_signal(hr, INPUT_LENGTH).T
        hr = hr.T
        lr_tensor = torch.from_numpy(lr.astype(np.float32))
        hr_tensor = torch.from_numpy(hr.astype(np.float32))
        return lr_tensor, hr_tensor
