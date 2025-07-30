from settings.basic_settings import BasicSettings

class SIGFormerSettings(BasicSettings):
    def __init__(self, data_name):
        super(SIGFormerSettings, self).__init__('sigformer', data_name)
        self.test_epoch = 5
        self.embs_dim = 64
        # transformer
        self.layer_num = 1
        # 
        self.sample_hop = 4
        # weight
        self.alpha = 0.4
        # 
        self.model_arc = "sse spe"
        self.sse_dim = 64 # 谱编码维度
    
    def Get_Search_Space(self) -> dict:
        search_space = {
                        
                        }
        return search_space

model_parameters:list = ['embs_dim', "layer_num",
                          "sse_dim", "sample_hop","model_arc",
                           "alpha" ]