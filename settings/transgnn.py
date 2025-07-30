from settings.basic_settings import BasicSettings

class TransGNNSettings(BasicSettings):
    def __init__(self, data_name):
        super(TransGNNSettings, self).__init__('transgnn', data_name)
        self.test_epoch = 8
        self.embs_dim = 64
        
        self.block_num = 1 # gcn + transformer
        self.trans_head_num = 2
        
    def Get_Search_Space(self) -> dict:
        search_space = {# "block_num":[1, 2, 3],
                        "trans_head_num":[1, 2, 4]}
        return search_space

model_parameters:list = ['embs_dim', "block_num", "trans_head_num"]