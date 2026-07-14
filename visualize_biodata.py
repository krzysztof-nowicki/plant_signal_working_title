"""Example script to load and visualize a single BioSignal NPZ.

This small script demonstrates loading a saved BioSignal and running
the convenience info/plot helpers.
"""
from data_classes.biosignal import BioSignal

biosignal = BioSignal.load(r"D:\Planty\data_converted\2nd Jan.npz")
# biosignal.plot_original()
# biosignal.listen()
biosignal.info()
biosignal.plot()
