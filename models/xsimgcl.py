from . import SimGCL
from .functions import Losses
from settings import XSimGCLSettings
from torch import nn, norm, rand, sign, cat, sparse, stack, mean, split


class XSimGCL(SimGCL):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: XSimGCLSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)
        self.noise_magnitude = params.noise_magnitude
        self.temp = params.temp
        self.cl_lambda = params.cl_lambda

    def encoder_predict(self):
        users, items = self.XSimGCN(False)
        return users, items
    
    def XSimGCN(self, with_noise_embs = True):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = cat([users_emb, items_emb]) #在零维对用户物品向量进行拼接

        embs = []
        embs_noise0 = None # embedding with noise
        
        for layer in range(self.layer_num):
            all_embs = sparse.mm(self.A_norm_sp, all_embs)
            embs.append(all_embs)
            if layer == 0 and with_noise_embs:
                embs_noise0 = all_embs + \
                                   SimGCL.generate_embedding_noise(all_embs, self.noise_magnitude)

        users, items = SimGCL.layer_combination(embs, self.user_num, self.item_num)

        if with_noise_embs:
            users_n0, items_n0 = split(embs_noise0, [self.user_num, self.item_num])
            return users, items, users_n0, items_n0
        else:# embs without noise
            return users, items
    
    def getEmbedding(self, uids, iids, jids):
        user_embs, item_embs, user_embs_n0, item_embs_n0 = self.XSimGCN(with_noise_embs=True)
        uid_embs = user_embs[uids]
        iid_embs = item_embs[iids]
        jid_embs = item_embs[jids]
        # noise embeddings
        users_emb_n0 = user_embs_n0[uids]
        items_emb_n0 = item_embs_n0[iids]
        return uid_embs, iid_embs, jid_embs, users_emb_n0, items_emb_n0
    
    def loss(self, data_batch):
        '''bpr loss , regularization loss'''
        uids, iids, jids = data_batch
        jids = jids.squeeze()
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
                "loss_xsimgcl":loss_total}
