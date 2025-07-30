from settings.basic_settings import BasicSettings

class SimGCLSettings(BasicSettings):
    def __init__(self, data_name):
        super(SimGCLSettings, self).__init__('simgcl', data_name)
        self.test_epoch = 8
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