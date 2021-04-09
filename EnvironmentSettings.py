class EnvironmentSettings:

    def __init__(self, env):
        if env == "HOME":
            self.set_home_paths()
        if env == "PAPERSPACE":
            self.set_paperspace_paths()
        if env == "AZURE":
            self.set_azure_paths()

    def set_home_paths(self):
        self.labels = "C:\MyFiles\Thesis\Project\data/train_labels.csv"
        self.train = "C:\MyFiles\Thesis\Project\data/train"
        self.val = "C:\MyFiles\Thesis\Project\data/test"
        self.checkpoint = "C:\MyFiles\Thesis\Project\outputs\checkpoints"
        self.checkpoint2 = "C:\MyFiles\Thesis\Project\outputs\checkpoints\2"
        self.batch_size = 128
        self.image_extension = "tif"
        
    def set_paperspace_paths(self):
        self.labels = "/notebooks/storage/cancer_detection/train_labels.csv"
        self.train = "/notebooks/storage/cancer_detection/train"
        self.val = "/notebooks/storage/cancer_detection/test"
        self.checkpoint = "/notebooks/storage/cancer_detection/Checkpoints"
        self.batch_size = 258
        self.image_extension = "tif"
        
    def set_azure_paths(self):
        self.labels = "C:/AI/data/train_labels.csv"
        self.train = "C:/AI/data/train"
        self.val = "C:/AI/data/test"
        self.checkpoint = "C:/AI/outputs/checkpoints"
        self.batch_size = 128
        self.image_extension = "tif"