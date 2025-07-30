from . import DataLightGCN
import tqdm
import random
import torch
from .tools import Tools
from torch import tensor, concat
from collections import defaultdict
from torch import svd_lowrank

class DataSIGFormer(DataLightGCN):
    def __init__(self, data_name):
        super(DataSIGFormer, self).__init__(data_name)
        self.path_types = []
    
    def get_neg_adj_table(self, adj_table:dict):
        nodes = list(adj_table.keys())
        neg_adj_table = defaultdict(list)
        for n in adj_table:
            for _ in range(len(adj_table[n])): # 等比例负采样
                neighbor = random.choice(nodes)
                while neighbor in adj_table[n]:
                    neighbor = random.randint(0, len(adj_table))
                neg_adj_table[n].append(neighbor)
        return neg_adj_table
    
    @property
    def neg_ratings(self):
        print("[neg_ratings]...")
        uids = self.ratings[:,0]
        neg_rate_table = self.Load_User_Neg_Dict()
        neg_ratings = []
        for uid in uids.tolist():
            if uid in neg_rate_table:
                neg_nei = random.choice(neg_rate_table[uid])
                neg_ratings.append([uid, neg_nei])
        return torch.tensor(neg_ratings)

    def SSE(self, alpha, sse_dim):
        """ Sign-aware Spectral Encoding """
        print("[SSE]...")
        pos_L = self.L
        neg_ratings = self.neg_ratings
        neg_A_norm = Tools.rating_to_A_norm(neg_ratings.T, self.user_num, self.item_num)
        
        neg_L = Tools.A_norm_to_L(neg_A_norm)
        
        L = 1/(1-alpha) * (pos_L - alpha*neg_L)

        return svd_lowrank(L, q = sse_dim)[0]

    def SPE(self, sample_hop):
        """Sign-aware Path Encoding"""
        adj_table = self.adj_table
        neg_adj_table = self.get_neg_adj_table(adj_table)
        pathes_ptypes = []
        for i in tqdm.tqdm(range(self.user_num+self.item_num), desc="[spe]"):
            ppt = Tools.random_walk_sigformer(i, sample_hop, adj_table, neg_adj_table)
            if ppt is not None:
                pathes_ptypes.append(tensor(ppt))
        pathes_ptypes = concat(pathes_ptypes, dim=0)
        return pathes_ptypes
    
    



    

    