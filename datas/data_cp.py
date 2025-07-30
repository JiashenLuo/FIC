from datas.data import DataBasic
from torch.utils.data import Dataset, DataLoader
from torch import arange

class DatasetCP(Dataset):
    def __init__(self, item_num, R):
        super(DatasetCP, self).__init__()
        self.item_num = item_num
        self.item_idxes = arange(self.item_num)
        self.RT = R.T
    
    def shuffle(self):
        pass

    def __len__(self):
        return self.item_num

    def __getitem__(self, idx):
        '''item, user_vec'''
        return self.item_idxes[idx], self.RT[idx]
    
class DataCP(DataBasic):
    def __init__(self, data_name):
        super(DataCP, self).__init__(data_name)
    
    def Get_Train_Loader(self, batch_size = 64):
        '''[item_idxes]
        '''
        dataset = DatasetCP(self.item_num, self.Load_R('train'))
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        return train_loader
    