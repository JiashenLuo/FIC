import torch.nn.functional as F
from torch import nn
import torch
from settings import TransGNNSettings
from numpy import save as npsave
from models import BPR, Losses, LightGCN


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, attn_mask=None, length_mask=None):
        N = x.shape[0]
        L = x.shape[1]
        
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=attn_mask
        )
        
        x = x + self.dropout(attn_output)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class TransGNN(BPR):
    def __init__(self, user_num, item_num, A_sparse,# data
                 # params
                 params:TransGNNSettings):
        super(TransGNN, self).__init__(user_num, item_num, params)
        self.params = params
        self.A_sparse = A_sparse
        # transformer
        self.user_transformer_encoder = TransformerEncoderLayer(d_model=params.embs_dim, 
                                                                num_heads=params.trans_head_num, 
                                                                dropout=0.0)
        self.item_transformer_encoder = TransformerEncoderLayer(d_model=params.embs_dim, 
                                                                num_heads=params.trans_head_num,
                                                                dropout=0.0)
     
    def model_to_cuda(self):
        super().model_to_cuda()
        self.A_sparse = self.A_sparse.cuda()
        self.user_transformer_encoder = self.user_transformer_encoder.cuda()
        self.item_transformer_encoder = self.item_transformer_encoder.cuda()

    def user_transformer_layer(self, embs, mask = None):
        assert len(embs.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embs.shape)
        if len(embs.shape) == 2:
            embs = embs.unsqueeze(dim=0)
            embs = self.user_transformer_encoder(embs, mask)
            embs = embs.squeeze()
        else:
            embs = self.user_transformer_encoder(embs, mask)
        
        return embs
    
    def item_transformer_layer(self, embs, mask=None):
        assert len(embs.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embs.shape)
        if len(embs.shape) == 2:
            embs = embs.unsqueeze(dim=0)
            embs = self.item_transformer_encoder(embs, mask)
            embs = embs.squeeze()
        else:
            embs = self.item_transformer_encoder(embs, mask)
        
        return embs

    def encoder_predict(self):
        trans_embs, user_embs, item_embs = self.encoder()
        return user_embs, item_embs
    
    def encoder(self):
        """forward part"""
        user_embs, item_embs = self.user_embs.weight, self.item_embs.weight
        embs = [torch.concat([user_embs, item_embs], dim=0)]     
        for i in range(self.params.block_num):
            user_embs_gcn, item_embs_gcn = LightGCN.LGCN(user_embs, item_embs, 
                                                         1, 
                                                         self.A_sparse)

            user_embs_trans = self.user_transformer_layer(user_embs_gcn)
            item_embs_trans = self.item_transformer_layer(item_embs_gcn)
            
            user_embs_trans += user_embs_gcn
            item_embs_trans += item_embs_gcn
        
            tmp_embs = torch.concat([user_embs_trans, item_embs_trans], dim=0)
            
            embs.append(tmp_embs)

        embs = sum(embs)
        user_embs = embs[:self.user_num]
        item_embs = embs[self.user_num:]
        return embs, user_embs, item_embs
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.squeeze().long()

        embs, user_embs, item_embs = self.encoder()
        # user_trans_embs, item_trans_embs = embs[:self.user_num], embs[self.user_num:]

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        jids_embs = item_embs[jids]
        loss_gcn = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)

        # uids_trans_embs = user_trans_embs[uids]
        # iids_trans_embs = item_trans_embs[iids]
        # jids_trans_embs = item_trans_embs[jids]
        # loss_trans = Losses.loss_BPR(uids_trans_embs, iids_trans_embs, jids_trans_embs)
        return {"loss_transgnn": loss_gcn}
    