from . import SimGCL, DNS
from . import XSimGCL
from .functions import Losses
from settings import DNSGCLSettings
from torch import nn, norm, rand, sign, cat, sparse, stack, mean, split
from torch import log, mul, exp, rand_like, diag
from torch.nn import functional as F


class DNSGCL(SimGCL):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: DNSGCLSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)
        self.cl_lambda = params.cl_lambda

    def getEmbedding(self, uids, iids, jids):
        user_embs, item_embs, user_embs_v1, item_embs_v1, user_embs_v2, item_embs_v2 = self.SimGCN(with_noise_embs=True)
        
        uid_embs = user_embs[uids]
        iid_embs = item_embs[iids]
        # dynamic negtive sampling
        jid_embs = DNS.Dynamic_Negative_Sampling(user_embs, item_embs, uids, jids)

        # noise embeddings
        users_emb_n1 = user_embs_v1[uids]
        items_emb_n1 = item_embs_v1[iids]
        users_emb_n2 = user_embs_v2[uids]
        items_emb_n2 = item_embs_v2[iids]
        return uid_embs, iid_embs, jid_embs, users_emb_n1, items_emb_n1, users_emb_n2, items_emb_n2
    
    def loss(self, data_batch):
        '''bpr loss , regularization loss'''
        uids, iids, jids = data_batch
        
        uid_embs, iid_embs, jid_embs, us_emb_n1, is_emb_n1, us_emb_n2, is_emb_n2 = \
                            self.getEmbedding(uids.long(), iids.long(), jids.long())
        '''bpr loss'''
        loss_bpr = Losses.loss_BPR(uid_embs, iid_embs, jid_embs)
        ''' cl (contrastive learning) loss'''
        user_cl_loss = Losses.InfoNCE(us_emb_n1, us_emb_n2, self.temp)
        item_cl_loss = Losses.InfoNCE(is_emb_n1, is_emb_n2, self.temp)

        loss_cl = user_cl_loss + item_cl_loss

        loss_total = loss_bpr + self.cl_lambda * loss_cl
        
        return {"loss_bpr":loss_bpr,
                "user_cl_loss":user_cl_loss,
                "item_cl_loss":item_cl_loss,
                "loss_dnsgcl":loss_total}
