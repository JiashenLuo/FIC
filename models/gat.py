from torch import nn
import torch
import torch.nn.functional as F
from . import BPR
from settings import GATSettings

class Graph_Attention_Layer(nn.Module):
    def __init__(self, attn_bs):
        super(Graph_Attention_Layer, self).__init__()
        self.attn_bs = attn_bs
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, embs, ratings, node_num):
        embs_w = embs

        attn_bs = self.attn_bs
        rate0, rate1 = ratings.T
        rate_num = ratings.shape[0]
        
        attn_values = []
        # 分批计算注意力分数
        for i in range(rate_num//attn_bs+1):
            r0_batch = rate0[i*attn_bs: min(rate_num, (i+1)*attn_bs)]
            r1_batch = rate1[i*attn_bs: min(rate_num, (i+1)*attn_bs)]
            # shape: (attn_bs, embs_dim * head_num)
            r0_embs, r1_embs = embs_w[r0_batch], embs_w[r1_batch]
            attn_score = (r0_embs * r1_embs).sum(-1)
            attn_values.append(attn_score)

        attn_values = torch.concat(attn_values, dim = 0)
        attn_sp = torch.sparse.FloatTensor(ratings.T, attn_values, (node_num, node_num))
        # 注意力得分softmax得到最终得分
        attn_sp = torch.sparse.softmax(attn_sp, dim = 1)
        attn_sp = attn_sp.detach()
        attn_embs = torch.sparse.mm(attn_sp, embs)
        return attn_embs

class GAT(BPR):
    def __init__(self, user_num, item_num, ratings,
                 params: GATSettings):
        super(GAT, self).__init__(user_num, item_num, params)
        self.node_num = self.user_num + self.item_num
        self.params = params

        self.gat_layers = []
        for _ in range(params.layer_num):
            self.gat_layers.append(Graph_Attention_Layer(params.attn_bs))
        self.ratings = ratings

    
    def model_to_cuda(self):
        super().model_to_cuda()
        self.ratings = self.ratings.cuda()
        for i in range(self.params.layer_num):
            self.gat_layers[i] = self.gat_layers[i].cuda()

    def encoder(self):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_emb = torch.cat([users_emb, items_emb]) #在零维对用户物品向量进行拼接

        embs = [all_emb]
        for i in range(self.params.layer_num):
            embs_gat = self.gat_layers[i](embs[-1], self.ratings, self.node_num)
            embs.append(embs_gat)

        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_num, self.item_num])
        return users, items