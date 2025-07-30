from settings.basic_settings import BasicSettings

class AHNSSettings(BasicSettings):
    def __init__(self, data_name):
        super(AHNSSettings, self).__init__('ahns', data_name)
        self.test_epoch = 3
        self.embs_dim = 64
        self.layer_num = 0
        self.neg_c = 10
        self.alpha = 1.0 # alpha in [0,1]
        self.beta = 0.1 # beta in [0,1]
        self.p = -1 # p in {-1,-2}
        
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['embs_dim', 'layer_num', "neg_c","alpha","beta","p"]