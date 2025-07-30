from settings.basic_settings import BasicSettings

class FICSettings(BasicSettings):
    def __init__(self, data_name):
        super(FICSettings, self).__init__('fic', data_name)
        self.test_epoch = 3
        self.embs_dim = 64
        self.layer_num = 2
        self.neg_c = 10
        
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['layer_num', "neg_c"]