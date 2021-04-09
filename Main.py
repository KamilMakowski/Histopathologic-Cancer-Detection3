from DataService import DataService
from KerasModels import KerasModels
from EnvironmentSettings import EnvironmentSettings
import matplotlib.pyplot as plt
ENV_SETTINGS = EnvironmentSettings("HOME")
data_service = DataService(ENV_SETTINGS.train, ENV_SETTINGS.val, ENV_SETTINGS.labels, "tif")

data_service.load(True)

keras_models = KerasModels(ENV_SETTINGS.checkpoint + "/NASNetMobile.h5", data_service, batch_size = ENV_SETTINGS.batch_size)

keras_models.train_NASNetMobile(2)
keras_models.show_final_history()
