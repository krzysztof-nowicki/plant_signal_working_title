"""Example script to load and visualize a single MusicSignal NPZ.

This small script demonstrates loading a saved MusicSignal and running
the convenience info/plot helpers.
"""

from data_classes.musicsignal import MusicSignal

musicsignal = MusicSignal.load(r"D:\Planty\music_data_converted\Hester - KYOTO [NCS Release].npz")
# musicsignal.plot_original()
# musicsignal.listen()
musicsignal.info()
musicsignal.plot()
