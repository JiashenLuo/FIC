from settings.basic_settings import BasicSettings

class LightGCLSettings(BasicSettings):
    def __init__(self, data_name):
        super(LightGCLSettings, self).__init__('lightgcl', data_name)
        self.embs_dim = 64
        self.svd_q = 5
        self.layer_num = 2
        self.dropout = 0.0
        self.test_epoch = 5
        self.embs_dim = 64
        self.layer_num = 2
        self.cl_lambda = 1e-7
        self.temp = 1.0
    
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()
    

model_parameters:list = ['embs_dim', 'layer_num',
                         'cl_lambda', 'temp']