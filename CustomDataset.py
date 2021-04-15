import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, pictures, labels, transform = None):
        self.images = pictures
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        if (self.transform):
            image = self.transform(self.images[index])
        else:
            image = self.images[index]
        return (image, self.labels[index])
        
        
