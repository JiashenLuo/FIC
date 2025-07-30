import torch
from torch import nn, diag, svd_lowrank, Tensor, gather, max
from torch import nn, sign, rand_like
from torch.nn import functional as F

class Losses:
    @staticmethod
    def loss_BPR(user_embs, pos_item_embs, neg_item_embs):
        '''BPR 损失'''
        pos_score = (user_embs * pos_item_embs).sum(dim=-1)
        neg_score = (user_embs * neg_item_embs).sum(dim=-1)
        loss = - (pos_score - neg_score).sigmoid().log().mean()
        return loss
    
    @staticmethod
    def InfoNCE(view1, view2, temp):
        '''InfoNCE loss'''
        view1, view2 = nn.functional.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / temp
        score = diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

class HardNegativeSampler:

    @staticmethod
    def FIC_sampling(user_embs, item_embs, uids, jids):
        '''动态负采样策略，返回采样后的难负样本的embedding
        shape: batch_size, 1'''
        uid_embs, jid_embs = user_embs[uids], item_embs[jids]
        uid_embs = uid_embs.unsqueeze(dim = 1) # [batch_size, 1, embs_dim]
        scores = (uid_embs * jid_embs).sum(dim = -1) # [batch_size, neg_c]

        indices = max(scores, dim = -1)[1]
        ng_jids = gather(jids, dim = 1, index = indices.unsqueeze(-1)).squeeze()

        return item_embs[ng_jids]

class cft_attention(nn.Module):
    def __init__(self, sample_hop = 1, model_arc = " "):
        super(cft_attention, self).__init__()
        self.model_arc = model_arc

        self.spec_lambda = torch.zeros(1)
        self.path_emb = nn.Embedding(2**(sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
    
    def model_to_cuda(self):
        self.spec_lambda = self.spec_lambda.cuda() # 谱域注意力系数
        self.path_emb = self.path_emb.cuda() # 路径编号对应的嵌入

    def forward(self, embs, SPE):
        embs = F.layer_norm(embs, normalized_shape=(embs.shape[-1],))
        embs_attn = self.attention(embs, embs, embs, SPE)
        return embs_attn
        
    def attention(self, q, k, v, SPE):
        attn_emb, attn_path = [], []

        pathes = SPE[:, :2]
        attn_emb = attention_batch(q, k, pathes[:, 0], pathes[:, 1], with_dim_norm=True)
        
        attn_emb = sparse_softmax(pathes.T, attn_emb, q.shape[0])

        if "spe" in self.model_arc:
            attn_path = self.path_emb(SPE[:, 2]).view(-1)
            attn_path = sparse_softmax(pathes.T, attn_path, q.shape[0])
            attn_emb = attn_emb + attn_path
        
        sp_graph = torch.sparse_coo_tensor(pathes.T, attn_emb, 
                                           torch.Size([q.shape[0], q.shape[0]]))
        sp_graph = sp_graph.coalesce()
        sp_graph.detach_()
        return torch.sparse.mm(sp_graph, v)

def embedding_add_noise(E: Tensor, noise_magnitude: float):
    """similar to the noise adding function in SimGCL"""
    random_noise = rand_like(E).cuda() # U(0,1) uniform distribute
    # sign符号函数，大于零的位置置1，否则置-1，目的是将e和deta放在一个超球面
    # 第二项为l2约束项
    E_noise = E + sign(E) * nn.functional.normalize(random_noise, dim=-1) * noise_magnitude
    return E_noise

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def SVD(Mat: Tensor, svd_k: int) -> tuple:
    """
    奇异值分解
    :param Mat: 待分解矩阵
    :param svd_k: 奇异值个数
    :return: U, V
    """
    print("SVDing...")
    U, S ,V = svd_lowrank(Mat, q = svd_k)
    return U, S, V

def attention_batch(q, k, path0, path1, with_dim_norm = False):
    '''分批求相似度得分'''
    attn_bs = 20480
    embs_dim = torch.tensor(q.shape[-1]) if with_dim_norm else torch.tensor(1.)
    attn, path_num = [], path0.shape[0]
    for i in range(path_num//attn_bs+1):
        p0_bs = path0[i*attn_bs:min(path_num, (i+1)*attn_bs)]
        p1_bs = path1[i*attn_bs:min(path_num, (i+1)*attn_bs)]
        # print(max(p0_bs), max(p1_bs))
        attn.append(torch.mul(q[p0_bs], k[p1_bs]).sum(dim=-1) * 1./torch.sqrt(embs_dim))
    return torch.concat(attn, dim=0)

def sum_norm(indices, values, n):
    # scatter_add: 将value的置依据索引添加到前面的张量中
    s = torch.zeros(n, device=values.device).scatter_add(0, indices[0], values)
    s[s == 0.] = 1.
    return values/s[indices[0]]

def sparse_softmax(indices, values, n):
    # clamp:限制张量上下界
    # n: q.shape[0]
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)


class Similarities:
    @staticmethod
    def sim_ip(user_embs, item_embs):
        """Inner Product Similarity"""
        return (user_embs * item_embs).sum(dim=-1)
