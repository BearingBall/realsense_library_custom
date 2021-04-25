import torch
import CustomDataset as cdset
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def preds_to_images(prediction):
    pred = prediction.log_softmax(dim=1).exp()
    _, indices = torch.max(pred,1)    
    return indices

def makeLoaders(folder, batch_size = 10, validation_fraction = 0.2, seed = 42):
    train_dataset = cdset.DepthDataset(folder, 
                           transform=transforms.Compose([
                           #transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           # Use mean and std for pretrained models
                           # https://pytorch.org/docs/stable/torchvision/models.html
                           #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           #      std=[0.229, 0.224, 0.225])                         
                       ])
                      )

    val_split = int(validation_fraction * len(train_dataset))
    
    indices = np.arange(1, len(train_dataset))
    
    np.random.seed(seed)
    np.random.shuffle(indices)

    val_indices, train_indices = indices[:val_split], indices[val_split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         sampler=val_sampler)
    return train_loader, val_loader


def resPlotter(loss, train, val):
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches((18,6))
    
    axs[0].plot(loss, label = "loss")
    axs[0].grid()
    axs[0].legend()
    
    axs[1].plot(train, label="train")
    axs[1].plot(val, label = "val")
    axs[1].grid()
    axs[1].legend()
    
    plt.show()