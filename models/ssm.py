from torch import nn, cosine_similarity
from . import BPR, LightGCN
from settings import SSMSettings


class SSM(LightGCN):
    def __init__(self, user_num, item_num, A_norm_sp,
                 params: SSMSettings):
        super(SSM, self).__init__(user_num, item_num, A_norm_sp,
                                  params)
        self.ssm_tau = params.ssm_tau

    @staticmethod
    def loss_ssm(uid_embs, iid_embs, jid_embs, tau):
        '''sampled softmax loss
        tau:温度系数'''
        pos_sim = cosine_similarity(uid_embs, iid_embs, dim = -1).mul(1/tau).exp()
        neg_sim = cosine_similarity(uid_embs.unsqueeze(1), jid_embs, dim = -1).mul(1/tau).exp()
        
        denominator = (pos_sim.unsqueeze(1) + neg_sim).sum(-1)
        loss_ssm = - (pos_sim/(denominator)).log().mean()
        return loss_ssm
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.squeeze().long()# 单一负样本则压缩第二个维度

        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        jids_embs = item_embs[jids]

        loss = self.loss_ssm(uids_embs, iids_embs, jids_embs, self.ssm_tau)/uids.shape[0]

        return {"loss_ssm": loss}
    