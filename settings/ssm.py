from settings.basic_settings import BasicSettings

class SSMSettings(BasicSettings):
    def __init__(self, data_name):
        super(SSMSettings, self).__init__('ssm', data_name)
        self.embs_dim = 64
        self.neg_c = 10
        self.layer_num = 2
        self.ssm_tau = 0.1 # 温度系数
    
    def Get_Search_Space(self) -> dict:
        search_space = {"ssm_tau":[i/10 for i in range(1, 11)]}
        return search_space

model_parameters:list = ['embs_dim','neg_c',"ssm_tau", "layer_num"]