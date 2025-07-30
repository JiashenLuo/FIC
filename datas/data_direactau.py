from datas.data import DataBasic
from random import shuffle
from torch.utils.data import Dataset, DataLoader

class Dataset_Ratings(Dataset):
    def __init__(self, ratings):
        super(Dataset_Ratings, self).__init__()
        """加载用户物品交互对"""
        self.ratings = ratings.tolist()
        self.shuffle()

    def shuffle(self):
        # rating shuffle
        shuffle(self.ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.ratings[idx][0]
        item_i = self.ratings[idx][1]
        return user, item_i
    
class DataDirectAU(DataBasic):
    def __init__(self, data_name):
        super(DataDirectAU, self).__init__(data_name)
    
    def Get_Train_Loader(self, batch_size = 64):
        ratings = self.Load_Data(use = 'train', type = 'tensor')
        dataset = Dataset_Ratings(ratings)
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        return train_loader
    
    