from . import LightGCN
from .functions import Losses
from settings import SimGCLSettings
from torch import nn, sign, cat, sparse, stack, mean, split
from torch import rand_like
from torch.nn import functional as F


class SimGCL(LightGCN):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: SimGCLSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)
        self.noise_magnitude = params.noise_magnitude
        self.temp = params.temp
        self.cl_lambda = params.cl_lambda

    def encoder_predict(self):
        users, items = self.SimGCN(False)
        return users, items
    
    @staticmethod
    def generate_embedding_noise(E, noise_magnitude):
        random_noise = rand_like(E).cuda() # U(0,1) uniform distribute
        # sign符号函数，大于零的位置置1，否则置-1，目的是将e和deta放在一个超球面
        # 第二项为l2约束项
        E_noise = E + sign(E) * nn.functional.normalize(random_noise, dim=-1) * noise_magnitude
        return E_noise
    
    @staticmethod
    def layer_combination(E_list, user_num, item_num):
        embs = stack(E_list, dim=1)
        light_out = mean(embs, dim=1)
        user_embs, item_embs = split(light_out, [user_num, item_num])
        return user_embs, item_embs
    
    def SimGCN(self, with_noise_embs = True):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = cat([users_emb, items_emb]) #在零维对用户物品向量进行拼接

        embs = []
        embs_noise1 = [] # embedding with noise
        embs_noise2 = [] # embedding with noise
        
        for layer in range(self.layer_num):
            all_embs = sparse.mm(self.A_norm_sp, all_embs)
            embs.append(all_embs)
            if with_noise_embs:
                embs_noise1.append(all_embs + 
                                   SimGCL.generate_embedding_noise(all_embs, self.noise_magnitude))
                embs_noise2.append(all_embs + 
                                   SimGCL.generate_embedding_noise(all_embs, self.noise_magnitude))
            # embs.append(all_embs)

        users, items = SimGCL.layer_combination(embs, self.user_num, self.item_num)

        if with_noise_embs:
            users_n1, items_n1 = SimGCL.layer_combination(embs_noise1, self.user_num, self.item_num)
            users_n2, items_n2 = SimGCL.layer_combination(embs_noise2, self.user_num, self.item_num)
            return users, items, users_n1, items_n1, users_n2, items_n2
        else:# embs without noise
            return users, items
    
    def getEmbedding(self, uids, iids, jids):
        user_embs, item_embs, user_embs_v1, item_embs_v1, user_embs_v2, item_embs_v2 = self.SimGCN(with_noise_embs=True)
        uid_embs = user_embs[uids]
        iid_embs = item_embs[iids]
        jid_embs = item_embs[jids]
        # noise embeddings
        users_emb_n1 = user_embs_v1[uids]
        items_emb_n1 = item_embs_v1[iids]
        users_emb_n2 = user_embs_v2[uids]
        items_emb_n2 = item_embs_v2[iids]
        return uid_embs, iid_embs, jid_embs, users_emb_n1, items_emb_n1, users_emb_n2, items_emb_n2
    
    def loss(self, data_batch):
        '''bpr loss , regularization loss'''
        uids, iids, jids = data_batch
        jids = jids.squeeze()
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
                "loss_simgcl":loss_total}
