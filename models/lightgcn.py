from . import BPR
from settings import LightGCNSettings
from torch import cat, sparse, stack, mean, split

class LightGCN(BPR):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params:LightGCNSettings):
        super(LightGCN, self).__init__(user_num, item_num, params)
        self.layer_num = params.layer_num
        self.A_norm_sp = A_norm_sp

    def model_to_cuda(self):
        super().model_to_cuda()
        self.A_norm_sp = self.A_norm_sp.cuda()

    def encoder(self):
        return LightGCN.LGCN(self.user_embs.weight, self.item_embs.weight, self.layer_num, 
                             self.A_norm_sp)

    def encoder_predict(self):
        return LightGCN.LGCN(self.user_embs.weight, self.item_embs.weight, self.layer_num, 
                             self.A_norm_sp)

    @staticmethod
    def LGCN(user_embs, item_embs, layer_num, A_norm_sp):
        """Light Weight Graph Convolution Filtering
        return user_embs, item_embs"""
        all_emb = cat([user_embs, item_embs]) #在零维对用户物品向量进行拼接

        embs = [all_emb]
        
        for layer in range(layer_num):
            all_emb = sparse.mm(A_norm_sp, all_emb)
            embs.append(all_emb)
            
        embs = stack(embs, dim=1)

        light_out = mean(embs, dim=1)
        users, items = split(light_out, [user_embs.shape[0], item_embs.shape[0]])
        return users, items
    