from . import DNS, XSimGCL
from .functions import Losses
from settings import DNSXSimGCLSettings


class DNSXSimGCL(XSimGCL):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: DNSXSimGCLSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)
        self.cl_lambda = params.cl_lambda

    def getEmbedding(self, uids, iids, jids):
        user_embs, item_embs, user_embs_n0, item_embs_n0 = self.XSimGCN(with_noise_embs=True)
        uid_embs = user_embs[uids]
        iid_embs = item_embs[iids]
        # dynamic negtive sampling
        jid_embs = DNS.Dynamic_Negative_Sampling(user_embs, item_embs, uids, jids)
        # noise embeddings
        users_emb_n0 = user_embs_n0[uids]
        items_emb_n0 = item_embs_n0[iids]
        return uid_embs, iid_embs, jid_embs, users_emb_n0, items_emb_n0
    
    def loss(self, data_batch):
        '''bpr loss , regularization loss'''
        uids, iids, jids = data_batch
        uid_embs, iid_embs, jid_embs, us_emb_n0, is_emb_n0 = \
                            self.getEmbedding(uids.long(), iids.long(), jids.long())
        '''bpr loss'''
        loss_bpr = Losses.loss_BPR(uid_embs, iid_embs, jid_embs)
        ''' cl (contrastive learning) loss'''

        user_cl_loss = Losses.InfoNCE(uid_embs, us_emb_n0, self.temp)
        item_cl_loss = Losses.InfoNCE(iid_embs, is_emb_n0, self.temp)

        loss_cl = user_cl_loss + item_cl_loss

        loss_total = loss_bpr + self.cl_lambda * loss_cl
        
        return {"loss_bpr":loss_bpr,
                "user_cl_loss":user_cl_loss,
                "item_cl_loss":item_cl_loss,
                "loss_dnsxsimgcl":loss_total}
    
