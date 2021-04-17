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
        if (self.transform):
            label = self.transform(self.labels[index])
        else:
            label = self.labels[index]    
            
            
        return (image, label)
        
        
