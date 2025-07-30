import torch.nn.functional as F
from torch import nn
import torch
from settings import SGFormerSettings
from . import BPR, LightGCN


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''
    def __init__(self, in_channels,
                 out_channels,
                 num_heads):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)

        self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation         node_num, num_heads, out_channels
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)# 除了头维度之外的维度进行矩阵乘法

        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones) 
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output
        

class SGFormer(BPR):
    def __init__(self, user_num, item_num, edges, A_norm_sp,# data
                 # params
                 params:SGFormerSettings):
        super(SGFormer, self).__init__(user_num, item_num, params)
        self.edges = edges
        self.A_norm_sp = A_norm_sp
        self.gcn_layer_num = params.gcn_layer_num
        self.feature_fusion = params.feature_fusion
        self.params = params
        # transformer
        self.trans_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(params.hidden_channels))
        for i in range(params.trans_layer_num):
            self.trans_convs.append(
                TransConvLayer(params.hidden_channels, params.hidden_channels, 
                               num_heads=params.trans_head_num))
            self.lns.append(nn.LayerNorm(params.hidden_channels))
        self.activation = F.relu

        # output f
        if self.feature_fusion == "cat":
            self.output_linear = nn.Linear(params.embs_dim * 2, params.embs_dim)
        else:
            self.output_linear = nn.Linear(params.embs_dim, params.embs_dim)
        
        # node
        self.node_embs = torch.nn.Embedding(user_num + item_num, params.embs_dim)
        nn.init.normal_(self.node_embs.weight, std=0.01)
 
    def model_to_cuda(self):
        self.A_norm_sp = self.A_norm_sp.cuda()
        self.node_embs = self.node_embs.cuda()
        for conv in self.trans_convs:
            conv = conv.cuda()
        for ln in self.lns:
            ln = ln.cuda()
        self.output_linear = self.output_linear.cuda()

    def trans_conv(self):
        layer_ = []

        # input MLP layer
        node_embs = self.node_embs.weight
        # simple global attention
        node_embs = self.lns[0](node_embs) # layer norm
        node_embs = self.activation(node_embs)

        # store as residual link
        layer_.append(node_embs)

        for i, conv in enumerate(self.trans_convs):
            # graph convolution with full attention aggregation
            node_embs = conv(node_embs, node_embs)
            node_embs = (node_embs + layer_[i]) / 2. # residual
            node_embs = self.lns[i + 1](node_embs) # layernorm
            node_embs = self.activation(node_embs)
            layer_.append(node_embs)

        return torch.split(node_embs, [self.user_num, self.item_num])
    
    def graph_encoder(self):
        user_embs, item_embs = torch.split(self.node_embs.weight, [self.user_num, self.item_num])
        user_embs, item_embs = LightGCN.LGCN(user_embs, item_embs, self.gcn_layer_num, self.A_norm_sp)
        return user_embs, item_embs

    def encoder(self):
        """forward part"""
        user_embs_trans, item_embs_trans = self.trans_conv()
        user_embs_graph, item_embs_graph = self.graph_encoder()
        # 子节点集划分为用户节点和物品节点
        if self.feature_fusion == 'add':
            g_weight = self.params.graph_weight
            user_embs = g_weight * user_embs_graph + (1-g_weight) * user_embs_trans
            item_embs = g_weight * item_embs_graph + (1-g_weight) * item_embs_trans
        else:
            user_embs = self.output_linear(torch.concat([user_embs_graph, user_embs_trans], dim = 1))
            item_embs = self.output_linear(torch.concat([item_embs_graph, item_embs_trans], dim = 1))

        return user_embs, item_embs

    