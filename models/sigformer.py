from models import BPR
from settings import SIGFormerSettings
from torch import nn
import torch
import torch.nn.functional as F
from .functions import sparse_softmax, attention_batch


class Attention_SIGFormer(nn.Module):
    def __init__(self, sample_hop, model_arc):
        super(Attention_SIGFormer, self).__init__()
        self.model_arc = model_arc

        self.spec_lambda = torch.zeros(1)
        self.path_emb = nn.Embedding(2**(sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
    
    def model_to_cuda(self):
        self.spec_lambda = self.spec_lambda.cuda() # 谱域注意力系数
        self.path_emb = self.path_emb.cuda() # 路径编号对应的嵌入

    def forward(self, embs, SSE, SPE):
        embs = F.layer_norm(embs, normalized_shape=(embs.shape[-1],))
        embs_attn = self.attention(
            embs, embs, embs,
            SSE,
            SPE)
        return embs_attn
        
    def attention(self, q, k, v, SSE, SPE):
        attn_emb, attn_spec, attn_path = [], [], []

        pathes = SPE[:, :2]

        attn_emb = attention_batch(q, k, pathes[:, 0], pathes[:, 1], with_dim_norm=True)
        
        if "sse" in self.model_arc:
            attn_spec = attention_batch(SSE, SSE, pathes[:, 0], pathes[:, 1])
            attn_emb_spec = attn_emb + self.spec_lambda * attn_spec
        else:
            attn_emb_spec = attn_emb

        attn_emb = sparse_softmax(pathes.T, attn_emb_spec, q.shape[0])

        if "spe" in self.model_arc:
            attn_path = self.path_emb(SPE[:, 2]).view(-1)
            attn_path = sparse_softmax(pathes.T, attn_path, q.shape[0])
            attn_emb = attn_emb + attn_path
        
        sp_graph = torch.sparse_coo_tensor(pathes.T, attn_emb, 
                                           torch.Size([q.shape[0], q.shape[0]]))
        sp_graph = sp_graph.coalesce()
        sp_graph.detach_()
        return torch.sparse.mm(sp_graph, v)

class SIGFormer(BPR):
    def __init__(self, user_num, item_num, SSE, SPE,
                 params: SIGFormerSettings):
        super(SIGFormer, self).__init__(user_num, item_num, params)
        self.params = params
        self.SSE = SSE
        self.SPE = SPE

        self.trans_layers = []
        for _ in range(params.layer_num):
            layer = Attention_SIGFormer(params.sample_hop
                            , params.model_arc)
            self.trans_layers.append(layer)
    
    def encoder(self):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = torch.cat([users_emb, items_emb]) 

        embs = [all_embs]
        for i in range(self.params.layer_num):
            all_embs = self.trans_layers[i](all_embs, self.SSE, self.SPE)
            embs.append(all_embs)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_num, self.item_num])

        return users, items
    
    def model_to_cuda(self):
        super().model_to_cuda()
        self.SSE = self.SSE.cuda()
        self.SPE = self.SPE.cuda()
        for i in range(self.params.layer_num):
            self.trans_layers[i].model_to_cuda()

    def save(self, path):
        params_dict = {"sse": self.SSE,
                       "spe": self.SPE,
                       "model": self.state_dict(),
                       }
        torch.save(params_dict, path)
    
    def load(self, path):
        params_dict = torch.load(path)
        self.SSE = params_dict["sse"]
        self.SPE = params_dict["spe"]
        self.load_state_dict(params_dict["model"])

        

   