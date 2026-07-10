"""WAV file conversion helpers.

Convert WAV PCM data into the project's BioSignal container. The
function keeps a broad argument list for compatibility with the other
converters and downstream code.
"""

import wave
import numpy as np
from biosignal import BioSignal


# Keep the converter API compatible with other converters; disable the
# too-many-arguments warning for this function.
# pylint: disable=too-many-arguments, too-many-positional-arguments
def convert_wav(file_path, channels=None, organism=None, species=None,
                recording_type=None, units=None, source=None):
    """
    Convert WAV file to BioSignal object.
    
    Args:
        file_path: Path to WAV file
        channels: List of channel names (optional)
        organism: Type of organism (plant, fungi, human, etc.)
        species: Species name
        recording_type: Type of recording (WAV, microphone, etc.)
        units: Signal units (V, mV, µV, etc.)
        source: Source/dataset name
    
    Returns:
        BioSignal object
    """
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        fs = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        audio_data = wav_file.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).T
        else:
            audio_array = audio_array.reshape(1, -1)

    if channels is None:
        channels = [f"Ch{i + 1}" for i in range(n_channels)]

    time = np.linspace(0, n_frames / fs, n_frames, dtype=np.float64)

    metadata = {
        'source_file': str(file_path),
        'sample_width': sample_width,
        'original_format': 'WAV'
    }

    return BioSignal(
        signal=audio_array.T,
        fs=fs,
        channels=channels,
        time=time,
        organism=organism or "unknown",
        species=species or "unknown",
        recording_type=recording_type or "WAV",
        units=units or "unknown",
        source=source or "WAV_recording",
        metadata=metadata
    )
