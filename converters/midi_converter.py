"""MIDI file conversion helpers.

Convert MIDI files into the project's MusicSignal container.
Uses pretty_midi for loading and synthesizing MIDI files.
"""

import numpy as np
from data_classes.musicsignal import MusicSignal, MusicMetadata

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None


# pylint: disable=too-many-arguments, too-many-positional-arguments
def convert_midi(file_path, channels=None, author=None, music_type=None,
                 length=None, source=None, fs=44100):
    """
    Convert MIDI file to MusicSignal object.

    Requires:
        pip install pretty_midi

    Args:
        file_path: Path to MIDI file
        channels: List of channel names (optional)
        author: Author/composer
        music_type: Music type
        length: Duration in seconds (auto-calculated if not provided)
        source: Source/dataset name
        fs: Sampling frequency used during synthesis

    Returns:
        MusicSignal object
    """
    if pretty_midi is None:
        raise ImportError(
            "pretty_midi is required. Install with: pip install pretty_midi"
        )

    try:
        midi = pretty_midi.PrettyMIDI(file_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MIDI file '{file_path}': {e}"
        ) from e

    # Synthesize MIDI to audio waveform
    audio = midi.synthesize(fs=fs).astype(np.float32)

    # Convert to (n_samples, n_channels)
    audio_array = audio.reshape(-1, 1)

    n_samples = audio_array.shape[0]
    time = np.linspace(
        0,
        n_samples / fs,
        n_samples,
        dtype=np.float64,
    )

    if channels is None:
        channels = ["Mono"]

    if length is None:
        length = midi.get_end_time()

    metadata = MusicMetadata(
        author=author or "unknown",
        music_type=music_type or "unknown",
        length=float(length),
        source=source or str(file_path),
        extra={
            "source_file": str(file_path),
            "original_format": "MIDI",
            "n_channels": 1,
            "n_samples": n_samples,
            "tempo_changes": len(midi.get_tempo_changes()[0]),
            "instruments": len(midi.instruments),
        },
    )

    return MusicSignal(
        values=audio_array,
        fs=int(fs),
        channels=channels,
        time=time,
        metadata=metadata,
    )
