from settings.basic_settings import BasicSettings

test_epoch_dict = {"frappe": 12}

class LightGCNSettings(BasicSettings):
    def __init__(self, data_name):
        super(LightGCNSettings, self).__init__('lightgcn', data_name)
        if data_name in test_epoch_dict:
            self.test_epoch = test_epoch_dict[data_name]
        else:
            self.test_epoch = 8
        self.embs_dim = 64
        self.layer_num = 2
    
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['embs_dim', 'layer_num']