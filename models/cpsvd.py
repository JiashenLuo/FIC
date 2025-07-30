from .abstract import AbstractCFRecommender
from math import ceil, floor
from torch import nn, sum, log, rand_like, svd_lowrank
from numpy import save as npsave

class Positive_Fit_Loss(nn.Module):
    def __init__(self, individuality=0.4) -> None:
        super().__init__()
        self.individuality = individuality # 个性度:投入兴趣区域数据的负类占正类的比例
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        # y_false prepare
        ng_rate = self.individuality * sum(y_true,dim=0)/sum(y_true+(1-y_true),dim=0)
        y_false = 1 - y_true
        rand = rand_like(y_true)
        mask = rand > ng_rate
        y_false[mask] = 0
        return sum(-1 * y_true * log(y_pred)- y_false * log(1 - y_pred))
    

class User_Preference_Net(nn.Module):
    def __init__(self, item_emb_dim, user_num, hidden_num):
        super().__init__()
        self.item_emb_dim = item_emb_dim
        self.user_num = user_num
        self.hidden_num = hidden_num
        self.net = self.get_user_net()

    def get_user_net(self):
        '''
        根据隐层数量生成对应的多层感知机
        '''
        item_user_gap=int((self.user_num-self.item_emb_dim)/(self.hidden_num+1))
        hiddens_dim=[(x+1)*item_user_gap for x in range(0,self.hidden_num)]
        net_dims=[self.item_emb_dim]+hiddens_dim+[self.user_num]
        net = nn.Sequential()
        for i in range(0,2 * (len(net_dims)-1),2):
            net.add_module('linear {}'.format(i),nn.Linear(net_dims[floor(i/2)],net_dims[ceil((i+1)/2)]))
            if i == (2*(len(net_dims)-1)-1-((2*(len(net_dims)-1)-1)%2)):
                net.add_module('sigmoid {}'.format(i+1),nn.Sigmoid())
            else:
                net.add_module('normalization {}'.format(i+1),nn.LayerNorm(net_dims[ceil((i+1)/2)]))
                net.add_module('sigmoid {}'.format(i+1),nn.Sigmoid())
        return net
    
    def forward(self, x):
        return self.net(x)


class CPSVD(nn.Module, AbstractCFRecommender):
    def __init__(self, embs_dim, user_num, hidden_num, individuality, R, R_norm):
        '''consensuse and persenality preference'''
        super().__init__()
        self.user_num = user_num
        self.item_vecs = CPSVD.svd(R_norm, embs_dim)[1]
        # 用户偏好网络
        self.UPN = User_Preference_Net(embs_dim, user_num, hidden_num)
        self.positive_fit_loss = Positive_Fit_Loss(individuality = individuality)
        self.R_T = R.T

    @staticmethod
    def svd(R_norm, svd_k):
        print("SVDing...")
        U,_ ,V = svd_lowrank(R_norm, q = svd_k)
        return U, V
    
    def model_to_cuda(self):
        self.item_vecs = self.item_vecs.cuda()
        self.positive_fit_loss = self.positive_fit_loss.cuda()
        # self.R_T = self.R_T.cuda()
        self.UPN = self.UPN.cuda()

    def loss(self, data_batch):
        '''regularization loss'''
        item_idx, user_vec = data_batch
        y_pred = self.forward(item_idx)
        pfl = self.positive_fit_loss(y_pred, user_vec)
        return {"positive_fit_loss": pfl}
    
    def forward(self, item_idxes = None):
        item_vecs = self.item_vecs
        output = self.UPN(item_vecs[item_idxes])
        return output
    
    def encoder(self):
        return super().encoder()
    
    def encoder_predict(self):
        return super().encoder_predict()
    
    def predict(self, topk, *args):
        item_vecs = self.item_vecs.cpu()
        pred_net = self.UPN.cpu()
        preds = pred_net(item_vecs).T
        self.UPN = self.UPN.cuda()
        self.item_vecs = self.item_vecs.cuda()
        train_R  = self.R_T.cpu().T
        preds = preds - train_R * 1e8
        
        _, predictions_idx = preds.topk(topk)
        return predictions_idx.cpu().numpy()

