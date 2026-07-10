"""Small helper script to inspect the converted BioSignal database.

This script is a convenience entry point used during development to
list files and print simple statistics from the conversion report.
"""

from biosignal_database import BioSignalDatabase


database = BioSignalDatabase(data_folder="./data_converted")

database.list_all()
print(database.get_statistics())
