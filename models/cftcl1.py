from models import DNS
from settings import CFTCL1Settings
import torch
from .functions import Losses, cft_attention, embedding_add_noise


class CFTCL1(DNS):
    def __init__(self, user_num, item_num, spe1, spe2,
                 params: CFTCL1Settings):
        super(CFTCL1, self).__init__(user_num, item_num, params)
        self.params = params
        self.spe1 = spe1
        self.spe2 = spe2
        self.cl_lambda = params.cl_lambda
        self.temp = params.temperature

        self.trans_layers = []
        for _ in range(1):
            layer = cft_attention()
            self.trans_layers.append(layer)

    def encoder_predict(self):
        users, items = self.encoder(contrastive_view=False)

        return users, items
    
    def encoder(self, contrastive_view = True):
        '''
        contrastive_view = False: inference by view 1
        '''
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = torch.cat([users_emb, items_emb]) 

        embs_view1 = [all_embs]
        for i in range(1):
            embs_view1.append(self.trans_layers[i](all_embs, self.spe1))
        
        embs1 = torch.mean(torch.stack(embs_view1, dim=1), dim=1)
        user_embs1, item_embs1 = torch.split(embs1, [self.user_num, self.item_num])
        if not contrastive_view:
            return user_embs1, item_embs1
        # constract view 2
        embs_view2 = [all_embs]
        for i in range(1):
            embs_view2.append(self.trans_layers[i](all_embs, self.spe2))
        embs2 = torch.mean(torch.stack(embs_view2, dim=1), dim=1)
        user_embs2, item_embs2 = torch.split(embs2, [self.user_num, self.item_num])
        return user_embs1, item_embs1, user_embs2, item_embs2
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.long()
        
        user_embs1, item_embs1, user_embs2, item_embs2 = self.encoder(contrastive_view = True)

        # bpr loss
        uids_embs1 = user_embs1[uids]
        iids_embs1 = item_embs1[iids]
        jids_embs = self.Dynamic_Negative_Sampling(user_embs1, item_embs1, uids, jids)
        loss_bpr = Losses.loss_BPR(uids_embs1, iids_embs1, jids_embs)/uids.shape[0]

        # contrastive loss
        uids_embs2 = user_embs2[uids]
        iids_embs2 = item_embs2[iids]
        cl_user = Losses.InfoNCE(uids_embs1, uids_embs2, self.temp)
        cl_item = Losses.InfoNCE(iids_embs1, iids_embs2, self.temp)
        cl_loss = cl_user + cl_item

        loss_total = loss_bpr+ self.cl_lambda * cl_loss

        return {"loss_bpr":loss_bpr,
                "cl_user":cl_user,
                "cl_item":cl_item,
                "loss_cftcl":loss_total}
    
    def save(self, path):
        torch.save({
            'model': self.state_dict(),
            'spe1': self.spe1,
            }, path)
    
    def load(self, load_path):
        params = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(params['model'])
        self.spe1 = params['spe1']
    
    def model_to_cuda(self):
        super().model_to_cuda()
        self.spe1 = self.spe1.cuda()
        self.spe2 = self.spe2.cuda()
        for i in range(1):
            self.trans_layers[i].model_to_cuda()
        