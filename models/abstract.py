import abc
import torch

class AbstractCFRecommender(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def model_to_cuda(self):
        pass

    @abc.abstractmethod
    def predict(self):
        '''生成预测矩阵R hat'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def encoder(self):
        '''获取最终的用户和物品embedding的权重
        即encoder'''
        raise NotImplementedError

    @abc.abstractmethod
    def encoder_predict(self):
        pass

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
    
    def load(self,load_path):
        # 加载模型
        self.load_state_dict(torch.load(load_path))
    
    @abc.abstractmethod
    def loss(self, data_batch)->dict:
        '''返回损失函数字典{key为损失函数名(str)：value为损失函数值(float)}
        当损失包含多项时，最后一个key代表需要优化的损失
        '''
        raise NotImplementedError
