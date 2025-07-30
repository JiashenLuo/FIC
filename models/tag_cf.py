from . import BPR, LightGCN
from settings import TAG_CFSettings


class TAG_CF(BPR):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params:TAG_CFSettings):
        super(TAG_CF, self).__init__(user_num, item_num, params)
        self.A_norm_sp = A_norm_sp
        self.params = params

    def model_to_cuda(self):
        super(TAG_CF, self).model_to_cuda()
        self.A_norm_sp = self.A_norm_sp.cuda()

    def encoder_predict(self):
        user_embs, item_embs = self.user_embs.weight, self.item_embs.weight
        user_embs, item_embs = LightGCN.LGCN(user_embs, item_embs, 
                                             self.params.message_pass_layer,
                                             self.A_norm_sp)

        return user_embs, item_embs
   
    