from settings.basic_settings import BasicSettings

class DNSMNSettings(BasicSettings):
    def __init__(self, data_name):
        super(DNSMNSettings, self).__init__('dnsmn', data_name)
        self.test_epoch = 3
        self.embs_dim = 64
        self.layer_num = 2
        self.neg_c = 200 # N 200, 500
        self.M = 10 # 1,2,3,4,5,10,20
        
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['layer_num', "neg_c", "M"]