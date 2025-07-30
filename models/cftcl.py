from models import DNS
from settings import CFTCLSettings
import torch
from .functions import Losses, cft_attention, embedding_add_noise


class CFTCL(DNS):
    def __init__(self, user_num, item_num, spe,
                 params: CFTCLSettings):
        super(CFTCL, self).__init__(user_num, item_num, params)
        self.params = params
        self.spe = spe
        self.noise_magnitude = params.noise_magnitude
        self.cl_lambda = params.cl_lambda
        self.temp = params.temperature

        self.trans_layers = []
        for _ in range(params.layer_num):
            layer = cft_attention(params.walk_len, 
                                  params.model_arc)
            self.trans_layers.append(layer)

    def encoder_predict(self):
        users, items = self.encoder(with_noise=False)
        return users, items
    
    def encoder(self, with_noise = True):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = torch.cat([users_emb, items_emb]) 

        embs = [all_embs]
        for i in range(self.params.layer_num):
            all_embs = self.trans_layers[i](all_embs, self.spe)
            embs.append(all_embs)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_embs, item_embs = torch.split(light_out, [self.user_num, self.item_num])
        if not with_noise:
            return user_embs, item_embs
        # add noise
        user_embs_n = embedding_add_noise(user_embs, self.noise_magnitude)
        item_embs_n = embedding_add_noise(item_embs, self.noise_magnitude)
        return user_embs, item_embs, user_embs_n, item_embs_n
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.long()
        
        user_embs, item_embs, user_embs_n, item_embs_n = self.encoder(with_noise = True)

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        # bpr loss
        jids_embs = self.Dynamic_Negative_Sampling(user_embs, item_embs, uids, jids)
        loss_bpr = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)/uids.shape[0]
        # contrastive loss
        uid_embs_n, iid_embs_n = user_embs_n[uids], item_embs_n[iids]
        cl_user = Losses.InfoNCE(uids_embs, uid_embs_n, self.temp)
        cl_item = Losses.InfoNCE(iids_embs, iid_embs_n, self.temp)
        cl_loss = cl_user + cl_item

        loss_total = loss_bpr + self.cl_lambda * cl_loss

        return {"loss_bpr":loss_bpr,
                "cl_user":cl_user,
                "cl_item":cl_item,
                "loss_cftcl":loss_total}
    
    def save(self, path):
        torch.save({
            'model': self.state_dict(),
            'spe': self.spe,
            }, path)
    
    def load(self, load_path):
        params = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(params['model'])
        self.spe = params['spe']
    
    def model_to_cuda(self):
        super().model_to_cuda()
        self.spe = self.spe.cuda()
        for i in range(self.params.layer_num):
            self.trans_layers[i].model_to_cuda()
        