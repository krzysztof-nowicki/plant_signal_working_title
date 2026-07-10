import numpy as np
from DataClasses import BioSignal

try:
    import pyedflib
except ImportError:
    pyedflib = None


def convert_edf(file_path, channels=None, organism=None, species=None,
                recording_type=None, units=None, source=None):
    """
    Convert EDF (European Data Format) file to BioSignal object.
    
    Requires pyedflib: pip install pyedflib
    
    Args:
        file_path: Path to EDF file
        channels: List of channel indices or names to extract (optional, all by default)
        organism: Type of organism
        species: Species name
        recording_type: Type of recording (EEG, LFP, EMG, etc.)
        units: Signal units
        source: Source/dataset name
    
    Returns:
        BioSignal object
    """
    if pyedflib is None:
        raise ImportError("pyedflib is required. Install with: pip install pyedflib")

    with pyedflib.EdfReader(file_path) as edf_file:
        n_channels = edf_file.signals_in_file
        fs = edf_file.samplefrequency(0)

        if channels is None:
            channels_to_read = list(range(n_channels))
        else:
            channels_to_read = []
            for ch in channels:
                if isinstance(ch, int):
                    channels_to_read.append(ch)
                else:
                    signal_labels = edf_file.getSignalLabels()
                    if ch in signal_labels:
                        channels_to_read.append(signal_labels.index(ch))

        signal_list = []
        channel_names = []
        units_list = []

        for ch_idx in channels_to_read:
            signal_data = edf_file.readSignal(ch_idx)
            signal_list.append(signal_data.astype(np.float32))
            label = edf_file.getSignalLabels()[ch_idx]
            channel_names.append(label)

            # Try to extract units from signal info
            try:
                phys_dim = edf_file.physical_dimension(ch_idx)
                units_list.append(phys_dim if phys_dim else "unknown")
            except:
                units_list.append("unknown")

        signal = np.array(signal_list).T
        n_samples = signal.shape[0]
        time = np.linspace(0, n_samples / fs, n_samples, dtype=np.float64)

        file_header = edf_file.getHeader()

    metadata = {
        'source_file': str(file_path),
        'patient_name': file_header.get('patient_name', ''),
        'recording_date': file_header.get('recording_date', ''),
        'units_per_channel': units_list,
        'original_format': 'EDF'
    }

    return BioSignal(
        signal=signal,
        fs=fs,
        channels=channel_names if channel_names else channels,
        time=time,
        organism=organism or "unknown",
        species=species or "unknown",
        recording_type=recording_type or "EDF",
        units=units or "µV",
        source=source or "EDF_recording",
        metadata=metadata
    )
