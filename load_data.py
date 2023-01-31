import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

class loadData:
    def __init__(self, args):
  
        self.preprocess=transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),  
        transforms.Normalize([0.5,0.5,0.5],
                             [0.5,0.5,0.5])
        ])
        self.batch_size = args.batch_size
    
        self.dataset = args.dataset
        if self.dataset == 'ImageNet':
            self.dataPath = args.imagenetPath
        elif self.dataset == 'CIFAR10':
            self.dataPath = args.cifar10Path
        elif self.dataset == 'CIFAR100':
            self.dataPath = args.cifar100Path

        self.create_paths(self.dataPath)


        
    @staticmethod
    def create_paths(path):
        if not os.path.exists(path):
            os.makedirs(path)


    def getDataloader(self):
        if self.dataset == 'ImageNet':
            imagenet_transforms=transforms.Compose([
                transforms.Resize((150,150)),
                transforms.ToTensor(),  
                transforms.Normalize([0.5,0.5,0.5],
                             [0.5,0.5,0.5])
            ])

            trainPath = os.path.join(self.dataPath, 'train')
            valPath = os.path.join(self.dataPath, 'val')

            train_loader=DataLoader(
                torchvision.datasets.ImageFolder(trainPath,transform=imagenet_transforms),
                batch_size=self.batch_size, shuffle=True
            )
            val_loader=DataLoader(
                torchvision.datasets.ImageFolder(valPath,transform=imagenet_transforms),
                batch_size=self.batch_size, shuffle=True
            )
        elif self.dataset == 'CIFAR10':
            
            cifar10_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    )
                ])
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(self.dataPath, train=True, download=True, 
                                             transform=cifar10_transfors),
                batch_size=self.batch_size, shuffle=True)

            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(self.dataPath, train=False, download=True, 
                                             transform=cifar10_transfors),
                batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader
    
    def getDictForInference(self, args):
        val_dict = {}
    






    


