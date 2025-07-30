from . import LightGCN
from torch import abs, min, gather
from .functions import Losses
from settings import AHNSSettings


class AHNS(LightGCN):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: AHNSSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)

    @staticmethod
    def Adaptive_Negative_Sampling(user_embs, item_embs, uids, iids, jids, beta, alpha, p):
        uid_embs, iid_embs, jid_embs = user_embs[uids], item_embs[iids], item_embs[jids]

        p_score = (uid_embs * iid_embs).sum(dim=-1).unsqueeze(1)
        n_score = (uid_embs.unsqueeze(1) * jid_embs).sum(dim=-1)
        '''adaptive negative sampling'''
        ahns_score = abs(n_score - beta*(p_score+alpha).pow(p+1))
        indices = min(ahns_score, dim = 1)[1].detach()
        neg_item = gather(jids, dim=1, index = indices.unsqueeze(-1)).squeeze()
        return item_embs[neg_item]
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.long()
        
        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        beta, alpha, p = self.params.beta, self.params.alpha, self.params.p
        jids_embs = AHNS.Adaptive_Negative_Sampling(user_embs, item_embs, uids, iids, jids, beta, alpha, p)

        loss = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)

        return {"loss_ahns": loss}
    
