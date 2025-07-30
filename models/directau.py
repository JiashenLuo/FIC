from models.bpr import BPR
from settings import DirectAUSettings
from torch import pdist
from torch.nn.functional import normalize

class DirectAU(BPR):
    def __init__(self, user_num, item_num, 
                 params: DirectAUSettings):
        super().__init__(user_num, item_num, params)
        self.gamma = params.gamma

    @staticmethod
    def alignment(x, y):
        x, y = normalize(x, dim=-1), normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = normalize(x, dim=-1)
        return pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
    
    def loss(self, data_batch) -> dict:
        uids, iids = data_batch
        uids = uids.long()
        iids = iids.long()
        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        
        align = self.alignment(uids_embs, iids_embs)
        uniform = (self.uniformity(uids_embs) + self.uniformity(iids_embs)) / 2

        return {"align":align, "uniform":uniform ,
                "loss_directau":align + self.gamma* uniform}