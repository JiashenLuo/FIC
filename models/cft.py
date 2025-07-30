from models import BPR, DNS
from settings import CFTSettings
import torch
from .sigformer import Attention_SIGFormer


class CFT(DNS):
    def __init__(self, user_num, item_num, sse, spe,
                 params: CFTSettings):
        super(CFT, self).__init__(user_num, item_num, params)
        self.params = params
        self.sse = sse
        self.spe = spe

        self.trans_layers = []
        for _ in range(params.layer_num):
            layer = Attention_SIGFormer(params.walk_len, 
                                  params.model_arc)
            self.trans_layers.append(layer)
    
    def encoder(self):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = torch.cat([users_emb, items_emb]) 

        embs = [all_embs]
        for i in range(self.params.layer_num):
            all_embs = self.trans_layers[i](all_embs, self.sse, self.spe)
            embs.append(all_embs)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_num, self.item_num])

        return users, items
    
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
        self.sse = self.sse.cuda()
        self.spe = self.spe.cuda()
        for i in range(self.params.layer_num):
            self.trans_layers[i].model_to_cuda()
        