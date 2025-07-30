from settings.basic_settings import BasicSettings

class CPSVDSettings(BasicSettings):
    def __init__(self, data_name):
        super(CPSVDSettings, self).__init__('cpsvd', data_name)
        self.batch_size = 1024
        self.embs_dim = 64
        self.hidden_num = 2
        self.individuality = 250 # individuality of loss function

    def Get_Search_Space(self) -> dict:
        search_space = {'individuality':[100, 150, 200, 250, 300],
                        'hidden_num':[1, 2, 3]}
        return search_space

model_parameters:list = ['embs_dim', 'hidden_num', 'individuality']