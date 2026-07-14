"""
Converters package for signal data format conversions.

This package provides converters for different file formats:
- WAV: convert_wav()
- CSV: convert_csv()
- EDF: convert_edf()
- MP3: convert_mp3()
"""

from converters.wav_converter import convert_wav
from converters.csv_converter import convert_csv
from converters.edf_converter import convert_edf
from converters.mp3_converter import convert_mp3, music_signal_to_mp3

__all__ = [
    'convert_wav',
    'convert_csv',
    'convert_edf',
    'convert_mp3',
    'music_signal_to_mp3',
]

