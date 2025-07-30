from . import DataLightGCN
from torch import concat, stack

class DataSGFormer(DataLightGCN):
    def __init__(self, data_name):
        super(DataSGFormer, self).__init__(data_name)
        self._edges = None

    @property
    def edges(self):
        '''edges shape: [num_edges, 2]'''
        ratings = self.ratings
        ratings[:, 1] += self.user_num
        nodes1, nodes2 = ratings.T
        
        # 构造无向图的边
        nodes1_new = concat([nodes1, nodes2], dim = 0)
        nodes2_new = concat([nodes2, nodes1], dim = 0)
        edges_undirected = stack([nodes1_new, nodes2_new], dim=0)
        return edges_undirected
    