from settings.basic_settings import BasicSettings

class BPRSettings(BasicSettings):
    def __init__(self, data_name):
        super(BPRSettings, self).__init__('bpr', data_name)
        self.embs_dim = 64
    
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['embs_dim']