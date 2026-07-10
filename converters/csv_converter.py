"""CSV converter helpers.

This module converts simple CSV exports into the project's BioSignal
container. It avoids heavy refactoring to keep behavior stable; the
function signature is intentionally wide to preserve downstream callers.
"""

import pandas as pd
import numpy as np
from biosignal import BioSignal


# The converter function is intentionally permissive in the number of
# keyword arguments it accepts (they are mapped into metadata).
# Disable pylint too-many-arguments warnings for backwards compatibility.
# pylint: disable=too-many-arguments, too-many-positional-arguments
def convert_csv(
        file_path,
        fs=1,
        channels=None,
        organism=None,
        species=None,
        recording_type=None,
        units=None,
        source=None
):
    """
    Convert CSV file to BioSignal object.

    Expected CSV format:
    HH:MM:SS, ch1, ch2, ch3, ...
    """

    df = pd.read_csv(file_path, header=None)
    time_strings = df.iloc[:, 0].astype(str)

    signal = (
        df.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    valid_rows = ~np.isnan(signal).any(axis=1)
    signal = signal[valid_rows]
    time_strings = time_strings[valid_rows].reset_index(drop=True)

    n_samples = signal.shape[0]

    time = np.arange(n_samples, dtype=np.float64) / fs

    if channels is None:
        channels = [f"Ch{i + 1}" for i in range(signal.shape[1])]

    metadata = {
        "source_file": str(file_path),
        "original_format": "CSV",
        "original_time": time_strings.tolist(),
        "n_rows": len(df),
        "n_samples": n_samples,
        "n_channels": signal.shape[1],
    }

    # Use the new structured BioSignal factory which accepts primitive
    # parts and builds Signal/Metadata internally.
    return BioSignal.from_parts(
        values=signal,
        fs=fs,
        channels=channels,
        time=time,
        organism=organism or "unknown",
        species=species or "unknown",
        recording_type=recording_type or "Differential CSV",
        units=units or "unknown",
        source=source or "CSV_data",
        extra=metadata,
    )
