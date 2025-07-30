
from . import DNS, LightGCN
from .functions import Losses
from .functions import HardNegativeSampler as HNS
from torch import gather
from . import LightGCN
from .functions import Losses
from .functions import HardNegativeSampler as HNS
from settings import FICSettings
'''
farther is closer
'''

class FIC(LightGCN):
    def __init__(self, user_num, item_num, A_norm_sp, 
                 params: FICSettings):
        super().__init__(user_num, item_num, A_norm_sp, params)

    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.long()
        
        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        jids_embs = HNS.FIC_sampling(user_embs, item_embs, uids, jids)

        loss = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)

        return {"loss_fic": loss}
    
