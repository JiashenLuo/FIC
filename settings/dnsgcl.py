from settings.basic_settings import BasicSettings

class DNSGCLSettings(BasicSettings):
    def __init__(self, data_name):
        super(DNSGCLSettings, self).__init__('dnsgcl', data_name)
        self.test_epoch = 5
        self.embs_dim = 64
        self.neg_c = 50
        self.layer_num = 2
        self.noise_magnitude = 0.2
        self.temp = 0.1
        self.cl_lambda = 0.0
    
    def Get_Search_Space(self) -> dict:
        search_space = {# "neg_c":[10, 50, 100, 500],
                "cl_lambda":[1.0, 1e-3, 1e-6, 1e-9, 0.0]
                }
        
        return search_space

model_parameters:list = ['embs_dim', 'neg_c', 'layer_num',
                         'noise_magnitude', 'temp', 'cl_lambda']