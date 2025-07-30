from settings.basic_settings import BasicSettings

class SGFormerSettings(BasicSettings):
    def __init__(self, data_name):
        super(SGFormerSettings, self).__init__('sgformer', data_name)
        self.test_epoch = 8
        self.embs_dim = 64
        # transformer
        self.hidden_channels = 64
        self.trans_layer_num = 1
        self.trans_head_num = 2
        # gcn
        self.gcn_layer_num = 2
        # feature fusion
        self.feature_fusion = "add" # "add" or "cat"
        self.graph_weight = 0.8
    
    def Get_Search_Space(self) -> dict:
        return super().Get_Search_Space()

model_parameters:list = ['embs_dim', "hidden_channels", "trans_layer_num",
                          "trans_head_num","gcn_layer_num", "feature_fusion", "graph_weight"]