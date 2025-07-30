from settings.basic_settings import BasicSettings

class TAG_CFSettings(BasicSettings):
    def __init__(self, data_name):
        super(TAG_CFSettings, self).__init__('tag_cf', data_name)
        self.embs_dim = 64
        self.message_pass_layer = 1
    
    def Get_Search_Space(self) -> dict:
        search_space = {
           'message_pass_layer': [1, 2, 3]
        }
        return search_space

model_parameters:list = ['embs_dim', "message_pass_layer"]