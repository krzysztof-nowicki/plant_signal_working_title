from PlantyProject.DataClasses import BioSignal

biosignal = BioSignal.load(r"D:\Planty\PlantyProject\data_converted\data.npz")
biosignal.info()
biosignal.plot()
# biosignal.plot_original()