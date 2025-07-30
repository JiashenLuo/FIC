from torch import diag, spmm

from .functions import Losses, SVD, sparse_dropout
from settings import LightGCLSettings
from . import BPR

class LightGCL(BPR):
    def __init__(self, user_num, item_num, adj_norm,
                 params: LightGCLSettings):
        super().__init__(user_num, item_num, params)
        self.adj_norm = adj_norm
        self.temp = params.temp
        self.layer_num = params.layer_num
        self.dropout = params.dropout
        self.cl_lambda = params.cl_lambda
        self.svd_q = params.svd_q
        self.U_T, self.V_T, self.U_mul_S, self.V_mul_S = self.svd_init()

    def svd_init(self):
        U, S, V = SVD(self.adj_norm, self.svd_q)
        print("U shape, S shape, V shape:", U.shape, S.shape, V.shape)
        U_mul_S = U @ (diag(S))
        V_mul_S = V @ (diag(S))
        return U.T, V.T, U_mul_S, V_mul_S
    
    def model_to_cuda(self):
        super().model_to_cuda()
        self.adj_norm = self.adj_norm.cuda()
        self.U_T = self.U_T.cuda()
        self.V_T = self.V_T.cuda()
        self.U_mul_S = self.U_mul_S.cuda()
        self.V_mul_S = self.V_mul_S.cuda()

    def encoder_predict(self):
        users, items = self.lightgcl(False)
        return users, items
    
    def global_gcn(self, user_embs, item_embs):
        # svd_adj propagation
        vt_ei = self.V_T @ item_embs
        uembs = (self.U_mul_S @ vt_ei)

        ut_eu = self.U_T @ user_embs
        iembs = (self.V_mul_S @ ut_eu)
        return uembs, iembs

    def local_gcn(self, user_embs, item_embs):
        uembs = spmm(sparse_dropout(self.adj_norm, self.dropout), 
                     item_embs)
        iembs = spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), 
                      user_embs)

        return uembs, iembs
    
    def lightgcl(self, train = True):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight

        local_user_emb = [users_emb]
        local_item_emb = [items_emb]
        global_user_embs = [users_emb]
        global_item_embs = [items_emb]
        
        for layer in range(self.layer_num):
            # gcn propagation
            uembs, iembs = self.local_gcn(local_user_emb[-1], 
                                          local_item_emb[-1])
            local_user_emb.append(uembs)
            local_item_emb.append(iembs)
            if train:
                # svd propagation
                g_uembs, g_iembs = self.global_gcn(global_user_embs[-1], 
                                                global_item_embs[-1])
                global_user_embs.append(g_uembs)
                global_item_embs.append(g_iembs)
            
        l_user_emb = sum(local_user_emb)
        l_item_emb = sum(local_item_emb)
        if train:
            g_user_embs = sum(global_user_embs)
            g_item_embs = sum(global_item_embs)
            return l_user_emb, l_item_emb, g_user_embs, g_item_embs
        else:
            return l_user_emb, l_item_emb
    
    def getEmbedding(self, uids, iids, jids):
        """
        return luembs, liembs, jid_lembs, uid_gembs, iid_gembs
        """
        luembs, liembs, guembs, giembs = self.lightgcl(True)
        # local uid embeddings
        
        # svd uid embeddings
        uid_gembs = guembs[uids]
        iid_gembs = giembs[iids]
        # negative items for bpr
        jid_lembs = liembs[jids]

        return luembs, liembs, jid_lembs, uid_gembs, iid_gembs
    
    def loss(self, data_batch):
        '''bpr loss , contrastive loss'''
        uids, iids, jids = data_batch
        jids = jids.squeeze()
        luembs, liembs, jid_lembs, uid_gembs, iid_gembs = \
                            self.getEmbedding(uids.long(), iids.long(), jids.long())
        uid_lembs, iid_lembs = luembs[uids], liembs[iids]
        
        '''bpr loss'''
        loss_bpr = Losses.loss_BPR(uid_lembs, iid_lembs, jid_lembs)
        '''lightgcl cl (contrastive learning) loss'''
        pos_user_cl = Losses.InfoNCE(uid_lembs, uid_gembs, self.temp)
        pos_item_cl = Losses.InfoNCE(iid_lembs, iid_gembs, self.temp)
        loss_cl = pos_user_cl + pos_item_cl


        loss_total = loss_bpr + self.cl_lambda * loss_cl
        
        return {"loss_bpr":loss_bpr,
                "loss_cl":loss_cl,
                "loss_lightgcl":loss_total}
