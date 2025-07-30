from settings.basic_settings import BasicSettings

class XSimGCLSettings(BasicSettings):
    def __init__(self, data_name):
        super(XSimGCLSettings, self).__init__('xsimgcl', data_name)
        self.test_epoch = 5
        self.embs_dim = 64
        self.layer_num = 2
        self.cl_lambda = 1e-3
        self.temp = 1.0
        self.noise_magnitude = 0.2
    
    def Get_Search_Space(self) -> dict:
        search_space = {'cl_lambda':[1, 1e-1, 1e-2, 1e-3],
                        'temp':[0.1, 0.4, 0.7, 1.0]}
        return search_space

model_parameters:list = ['embs_dim', 'layer_num',
                         'cl_lambda', 'temp', 'noise_magnitude']