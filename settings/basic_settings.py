import json
from os import path, makedirs
import abc 

class BasicSettings(metaclass = abc.ABCMeta):
    def __init__(self, model_name, data_name):
        self.batch_size:int = 4096
        self.lr:float = 1e-3
        self.epoches = 1500
        self.test_epoch = 10
        self.topk = 40
        self.device = "cuda:0"
        self.model_name = model_name
        self.data_name = data_name

    @abc.abstractmethod
    def Get_Search_Space(self)->dict:
        '''超参数搜索空间'''
        return dict()

    def Save(self, save_path):
        '''将训练设置以及模型结构参数保存到json文件中'''
        with open(save_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 2)



