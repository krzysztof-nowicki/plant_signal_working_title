"""Example script to load and visualize a single BioSignal NPZ.

This small script demonstrates loading a saved BioSignal and running
the convenience info/plot helpers.
"""

from data_classes.biosignal import BioSignal


biosignal = BioSignal.load(r"D:\Planty\data_converted\data.npz")
biosignal.info()
biosignal.plot()  # biosignal.plot_original()
