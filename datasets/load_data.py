import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transform

class loadData:
    def __init__(self, args):
  
    self.preprocess=transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),  
        transforms.Normalize([0.5,0.5,0.5],
                             [0.5,0.5,0.5])
    ])
    
    self.dataset = args.dataset
    if self.dataset == 'ImageNet':
        self.dataPath = args.imagenetPath
    elif self.dataset == 'CIFAR10':
        self.dataPath = args.cifar10Path
    elif self.dataset == 'CIFAR100':
        self.dataPath = args.cifar100Path


    def getDataloader(self, args):
        trainPath = os.path.join(self.dataPath, 'train')
        valPath = os.path.join(self.dataPath, 'val')

        train_loader=DataLoader(
            torchvision.datasets.ImageFolder(trainPath,transform=transformer),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader=DataLoader(
            torchvision.datasets.ImageFolder(valPath,transform=transformer),
            batch_size=args.batch_size, shuffle=True
        )

        return train_loader, val_loader
    
    def getDictForInference(self, args):
        val_dict = {}
    






    


