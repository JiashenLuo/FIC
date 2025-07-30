from settings.basic_settings import BasicSettings

class DirectAUSettings(BasicSettings):
    def __init__(self, data_name):
        super(DirectAUSettings, self).__init__('directau', data_name)
        self.test_epoch = 1
        self.batch_size = 256
        self.embs_dim = 64
        self.gamma = 1.0

    def Get_Search_Space(self) -> dict:
        search_space = {"gamma": [0.01,  0.1, 1.0, 10.0]}
        return search_space


model_parameters:list = ['embs_dim', 'gamma']