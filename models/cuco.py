from . import LightGCN
from torch import gather, topk, randint, save ,device, load
from .functions import Losses
from settings import DNSMNSettings


class CuCo(LightGCN):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: DNSMNSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)
        self.step = 1

    def encoder_predict(self):
        self.step += 1
        return LightGCN.LGCN(self.user_embs.weight, self.item_embs.weight, self.layer_num, 
                             self.A_norm_sp)

    # def save(self, path):
    #         save({
    #         'model': self.state_dict(),
    #         'params': self.params,
    #         }, path)
    
    # def load(self, load_path):
    #     model_dict = load(load_path, map_location=device('cpu'))
    #     self.load_state_dict(model_dict['model'])
    #     self.params = model_dict['params']

    @staticmethod
    def CuCo_sampling(user_embs, item_embs, uids, jids, M, t, T):
        '''
        M: memory bank size
        t: step
        T: total steps'''
        uid_embs, jid_embs = user_embs[uids], item_embs[jids]
        batch_size = uid_embs.shape[0]
        n_score = (uid_embs.unsqueeze(1) * jid_embs).sum(dim=-1)
        '''cuco sampling'''
        m = max(1, int(M * (t/T)))
        indices = topk(n_score, m, dim=1)[1].detach()
        rand_indices = randint(0, m, (batch_size, )).to(uid_embs.device)
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
        M, t, T = self.params.neg_c, self.step, self.params.epoches
        jids_embs = CuCo.CuCo_sampling(user_embs, item_embs, uids, jids, M, t, T)

        loss = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)

        return {"loss_cuco": loss}
    
