from settings.basic_settings import BasicSettings
"""
通过加噪声的方式构造对比视图，并使用contrastive learning进行训练
"""
class CFTCLSettings(BasicSettings):
    def __init__(self, data_name):
        super(CFTCLSettings, self).__init__('cftcl', data_name)
        self.test_epoch = 5
        self.embs_dim = 64
        self.layer_num = 1
        self.neg_c = 50
        # 
        self.walk_len = 4
        self.path_num = 40
        self.with_neg_edges = True
        self.model_arc = ' '
        # contrastive learning
        self.noise_magnitude = 1e-4
        self.temperature = 1.0
        self.cl_lambda = 1e-6

    def param_init(self):
        self.neg_c_map = {"ml_100k": 5, "ml_1m": 10, "douban_book": 50
                          , "yelp2018": 50, "gowalla": 50}
        self.path_num_map = {"ml_100k": 60, "ml_1m": 80, "douban_book": 40
                             , "yelp2018": 16, "gowalla": 16}
        self.model_arc_map = {"ml_100k": " ", "ml_1m": "spe sse", "douban_book": "sse"
                              , "yelp2018": "sse", "gowalla": ' '}
    
    def Get_Search_Space(self) -> dict:
        search_space = {
                        "layer_num":[2, 3],
                        }
        return search_space

model_parameters:list = ['embs_dim', "layer_num",
                          "sse_dim", "walk_len","path_num","sse_type",
                          "model_arc"
                          ]