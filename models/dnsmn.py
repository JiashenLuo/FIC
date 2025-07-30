from . import LightGCN
from torch import min, gather, topk, randint
from .functions import Losses
from settings import DNSMNSettings


class DNSMN(LightGCN):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: DNSMNSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)

    @staticmethod
    def Dynamic_MN_Sampling(user_embs, item_embs, uids, jids, M):
        uid_embs, jid_embs = user_embs[uids], item_embs[jids]
        batch_size = uid_embs.shape[0]
        n_score = (uid_embs.unsqueeze(1) * jid_embs).sum(dim=-1)
        '''dynamic mn sampling'''
        indices = topk(n_score, M, dim=1)[1].detach()
        rand_indices = randint(0, M, (batch_size, )).to(uid_embs.device)
        result_indices = gather(indices, dim=1, index = rand_indices.unsqueeze(1)).squeeze()
        neg_item = gather(jids, dim=1, index = result_indices.unsqueeze(-1)).squeeze()
        return item_embs[neg_item]
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.long()
        
        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        jids_embs = DNSMN.Dynamic_MN_Sampling(user_embs, item_embs, uids, jids, self.params.M)

        loss = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)

        return {"loss_dnsmn": loss}
    
