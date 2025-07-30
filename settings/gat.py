from settings.basic_settings import BasicSettings

class GATSettings(BasicSettings):
    def __init__(self, data_name):
        super(GATSettings, self).__init__('gat', data_name)
        self.embs_dim = 64
        self.layer_num = 1
        
        self.attn_bs = 10240 # 计算注意力分数的batch_size
    
    def Get_Search_Space(self) -> dict:
        search_space = {"layer_num":[1, 2, 3]}
        return search_space

model_parameters:list = ['embs_dim', 'layer_num']