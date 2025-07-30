from settings.basic_settings import BasicSettings
test_epoch_d = {"gowalla": 10}
class DNSSettings(BasicSettings):
    def __init__(self, data_name):
        super(DNSSettings, self).__init__('dns', data_name)
        self.test_epoch = 5
        self.layer_num = 2
        self.embs_dim = 64
        self.neg_c = 2
    
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['embs_dim', 'neg_c']