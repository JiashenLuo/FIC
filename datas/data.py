import os
import abc
import time
import tqdm
import torch
import numpy as np
import pandas as pd
from math import floor
import scipy.sparse as sp
from .tools import Tools
import warnings
import pickle
warnings.filterwarnings("ignore")


class DataBasic(metaclass = abc.ABCMeta):
    def __init__(self, data_name):
        '''Data基类'''
        self.data_name = data_name
        self.__data_root = os.path.join('datas', 'datasets', data_name)
        self.__df_root = os.path.join(self.__data_root, 'ratings.csv')

        self.data_df = pd.read_csv(self.__df_root)
        self.ratings = torch.tensor(self.data_df.values)
        self.num_rating = len(self.data_df)
        self.user_num = len(self.data_df['userid'].value_counts().keys())
        self.item_num = len(self.data_df['itemid'].value_counts().keys())
        # 输出相关信息
        print(f"[Dataset]:{data_name}")
        print(f"[#User]:{self.user_num}|[#Item]:{self.item_num}|[#Rating]:{self.num_rating}|\
                [Density]:{self.num_rating/(self.user_num*self.item_num):.4f}")
        # 未进行分割则自动分割数据
        if not os.path.exists(os.path.join('datas', "datasets",self.data_name,'ratings_test.csv')):
            print(f"第一次读取，自动分割，随机分割种子为{2024}")
            np.random.seed(2024)  
            self.Split_Data()

    @property
    def data_root(self):
        return self.__data_root
    
    @property
    def adj_table(self):
        '''图的邻接表
        '''
        data_path = os.path.join(self.__data_root, f'ratings_train.csv')
        data_df = pd.read_csv(data_path)
        adj_table = self.Rating_df_To_Rating_dict(data_df, adj=True)
        return adj_table

    def Load_Data(self, use = 'train', type = "dataframe"):
        '''从硬盘加载交互数据
        use: train, valid, test
        type: dataframe, dict, tensor
        '''
        data_path = os.path.join(self.__data_root, f'ratings_{use}.csv')
        data_df = pd.read_csv(data_path)
        if type == "dataframe":
            return data_df
        elif type == "dict":
            return self.Rating_df_To_Rating_dict(data_df)
        elif type == "tensor":
            return torch.tensor(data_df.values)
        
    def Rating_df_To_Rating_dict(self, RDf, adj = False):
        ratings = torch.tensor(RDf.values).T
        
        if not adj:# uid: items[tensor type]
            # 磁盘读取图的邻接表
            if os.path.exists(os.path.join(self.__data_root, 'rating_table.pkl')):
                print(f"Loading rating_table from disk")
                with open(os.path.join(self.__data_root, 'rating_table.pkl'), 'rb') as f:
                    rating_dict = pickle.load(f)
                return rating_dict
            values = torch.ones(ratings.shape[1])
            R_sparse = torch.sparse_coo_tensor(ratings, values, (self.user_num, self.item_num))
            rating_dict = dict()

            for uid in tqdm.tqdm(range(self.user_num), desc = '[DTD]'): # dataframe to dict
                uid_ratings = R_sparse[uid].coalesce().indices().squeeze() # 用户uid交互过的物品

                rating_dict[uid] = uid_ratings
            
            # 存储交互邻接表
            with open(os.path.join(self.__data_root, 'rating_table.pkl'), 'wb') as f:
                pickle.dump(rating_dict, f)
            return rating_dict
        else:# nodeid: neighbors
            # 磁盘读取图的邻接表
            if os.path.exists(os.path.join(self.__data_root, 'adj_table.pkl')):
                print(f"Loading adj_table from disk")
                with open(os.path.join(self.__data_root, 'adj_table.pkl'), 'rb') as f:
                    adj_table = pickle.load(f)
                return adj_table
            
            ratings[1,:] = ratings[1,:] + self.user_num
            ratings_reverse = torch.concat([ratings[1,:].reshape(1, -1), 
                                            ratings[0,:].reshape(1, -1)], dim = 0)
            ratings = torch.concat([ratings, ratings_reverse], dim = 1)
            values = torch.ones(ratings.shape[1])
            adj_sparse = torch.sparse_coo_tensor(ratings, values, 
                                               (self.user_num + self.item_num, 
                                                self.user_num + self.item_num))
            adj_table = dict()

            for nid in tqdm.tqdm(range(self.user_num+self.item_num), desc = '[DTD]'):
                neighbors = adj_sparse[nid].coalesce().indices().squeeze() # 用户uid交互过的物品
                nei_list = neighbors.tolist() if type(neighbors.tolist()) != int \
                                                else [neighbors.tolist()]
                if nei_list:
                    adj_table[nid] = set(nei_list)
            # 存储图的邻接表
            with open(os.path.join(self.__data_root, 'adj_table.pkl'), 'wb') as f:
                pickle.dump(adj_table, f)

            return adj_table

    
    def Rating_dict_To_Rating_df(self, RD):
        '''交互的字典形式转交互的dataframe形式'''
        ratings = []
        for uid, items in RD.items():
            items = torch.tensor(items).view(-1, 1)
            uids = uid * torch.ones_like(items)
            uid_rating = torch.concat([uids, items], dim = 1)
            ratings.append(uid_rating)
        ratings = torch.concat(ratings, dim = 0)
        return pd.DataFrame(ratings.tolist())
    
    def Split_Data(self, split_rate = [0.7, 0.1, 0.2]):
        '''分割数据集，默认[训练:验证:测试=7:1:2]

        设置随机数可保证特定分割
        '''
        print(f"Dataset Split Rate {split_rate}")
        # 构造稀疏交互矩阵
        ratings = torch.tensor(self.data_df.values).T
        values = torch.ones(ratings.shape[1])
        R_sparse = torch.sparse_coo_tensor(ratings, values, (self.user_num, self.item_num))
        train_dict, valid_dict, test_dict = dict(),dict(),dict()
        # 将用户的交互数据按分割比进行切分
        for uid in tqdm.tqdm(range(self.user_num), desc="Data Spliting"):
            uid_ratings = R_sparse[uid].coalesce().indices().squeeze().view(-1) # 用户uid交互过的物品
            # print(uid_ratings)
            n_rate = uid_ratings.shape[0]
            if n_rate<3:
                continue
            rating_set = set(uid_ratings.tolist())
            # 交互数据过少，则随机测试分一个物品，验证分一个物品
            if uid_ratings.shape[0]*split_rate[1]<1:
                valid_idx = np.random.choice(list(rating_set), size = 1)
                rating_set -= set(valid_idx) # 去除已选
                test_idx = np.random.choice(list(rating_set), size = 1)
                rating_set -= set(test_idx) # 去除已选
                train_idx = list(rating_set)
            else: # 否则按正常切分比进行划分(向下取整)
                valid_idx = np.random.choice(list(rating_set), size = floor(split_rate[1]*n_rate) )
                rating_set -= set(valid_idx) # 去除已选
                test_idx = np.random.choice(list(rating_set), size = floor(split_rate[2]*n_rate))
                rating_set -= set(test_idx) # 去除已选
                train_idx = list(rating_set)
                train_dict[uid] = train_idx
                valid_dict[uid] = valid_idx
                test_dict[uid] = test_idx
                
        ratings_train = self.Rating_dict_To_Rating_df(train_dict)
        ratings_valid = self.Rating_dict_To_Rating_df(valid_dict)
        ratings_test = self.Rating_dict_To_Rating_df(test_dict)
        # 数据保存
        ratings_train.to_csv(os.path.join(self.__data_root,'ratings_train.csv'), index=False, header=['uid', 'iid'])
        ratings_valid.to_csv(os.path.join(self.__data_root,'ratings_valid.csv'), index=False, header=['uid', 'iid'])
        ratings_test.to_csv(os.path.join(self.__data_root,'ratings_test.csv'), index=False, header=['uid', 'iid'])
    
    def Load_R(self, use = 'train', R_type = 'dense_tensor'):
        '''
        use: train/test/valid/None(all)
        type: sparse_tensor, dense_tensor, sp_lil, sp_csr
        '''
        data_df = self.Load_Data(use = use)
        users = torch.tensor(data_df['uid'].values)
        items = torch.tensor(data_df['iid'].values)

        indices = torch.concat([users.view(1,-1),items.view(1,-1)],dim = 0)
        values = torch.ones_like(users)
        
        R = sp.csr_matrix((values, indices)
                                    ,(self.user_num,self.item_num))
        if R_type == 'sp_lil':
            R = R.tolil()
            return R
        elif R_type == 'sp_dok':
            return R.todok()
        elif R_type == 'dense_tensor':
            R = Tools.sp_mat_to_sp_tensor(R)
            R = R.to_dense()
            return R
        elif R_type == 'sp_csr':
            return R
        else: # sparse tensor 
            R = Tools.sp_mat_to_sp_tensor(R)
            return R
    
    def Load_R_norm(self, use = 'train'):
        '''
        use: train/test/valid/None(all)
        dense: dense tensor or sparse tensor(coo_mat)
        '''
        R = self.Load_R('train', "sp_csr")
        # normalize
        R = R.todok()

        rowsum = np.array(R.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        R_norm = d_mat.dot(R)
        
        colsum = np.array(R.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        R_norm = R_norm.dot(d_mat)

        R_norm = R_norm.tocsr()
        return Tools.sp_mat_to_sp_tensor(R_norm)
    
    def Get_User_Rated_Items(self)->dict:
        '''获取用户交互过的物品键值对'''
        R = self.Load_R("train", R_type='dense_tensor')
        user_rated_items = dict()
        for uid in tqdm.tqdm(range(self.user_num), desc = 'user_rated_items'):
            user_rated_items[uid] = R[uid].nonzero().squeeze()
        return user_rated_items
    
    @abc.abstractmethod
    def Get_Train_Loader(self):
        pass

    def Get_Test_Labels(self):
        test_coo_R = self.Load_R(use = 'test', R_type = 'sparse_tensor')
        test_labels = [test_coo_R[i].coalesce().indices().view(-1).tolist() 
                       for i in range(self.user_num)]
        return test_labels



if __name__ == "__main__":
    data = DataBasic("frappe")

    