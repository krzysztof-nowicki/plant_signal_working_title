from PlantyProject.biosignal_database import BioSignalDatabase

database = BioSignalDatabase(data_folder="./PlantyProject/data_converted")

database.list_all()
print(database.get_statistics())