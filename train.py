import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from load_data import loadData
from Bcos_nets import resNet34
from baseline_models import baseResNet34
import torch.optim as optim
from utils import plot_losses
import argparse
import os
from tqdm import tqdm
from args_parameters import getArgs
import numpy as np
from torch.autograd import Variable
from utils import AddInverse
from teste import testeResNet34


class trainingBcos:
    def __init__(self, args):
        self.loader = loadData(args)
        self.model = testeResNet34()
        self.create_paths(args.save_ckpt, args.save_losses)

    @staticmethod
    def create_paths(ckpt_path, losses_path):
        print('*preparing for traing*')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        print(losses_path)
        if not os.path.exists(losses_path):
            os.makedirs(losses_path)
            print(ckpt_path, losses_path)


    def training(self, args):
        print('____________________________________')
        print('|                                  |')
        print('|     WELCOME TO BCOS TRAINING     |')
        print('|                                  |')
        print('____________________________________')
        
        #LOADING DATA
        train_dataloader, val_dataloader = self.loader.getDataloader()

        #TRAINING PARAMETERS
        lr = 3e-4
        iter_num = 0
        patience = 10
        patience_counter = 0
        best_val_loss = 1e9
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters())
        all_train_loss = []
        all_val_loss = []

        self.model.to(args.device)
        print(self.model)



        #############################
        #       TRAINING LOOP       #
        #############################

        print(f' STARTING TRAINING WITH {args.model_name} FOR {args.dataset}')

        for epoch in range(args.epochs):
            epoch_val_losses = []
            epoch_train_losses = []

            self.model.train()
            for imgs, labels in tqdm(train_dataloader):
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                #imgs = Variable(AddInverse()(imgs), requires_grad=True)
                labels = F.one_hot(labels, num_classes=10)

                output = self.model(imgs)
                optimizer.zero_grad()
                loss = criterion(output, labels.float())
                loss.backward()
                optimizer.step()
                iter_num += 1
                epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    
            self.model.eval()
            for imgs, labels in val_dataloader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                #imgs = Variable(AddInverse()(imgs), requires_grad=True)
                labels = F.one_hot(labels, num_classes=10)
                output = self.model(imgs)
                loss = criterion(output, labels.float())
                epoch_val_losses.append(loss.detach().cpu().numpy())

            train_loss, val_loss = np.mean(epoch_train_losses), np.mean(epoch_val_losses)
 
            #Early Stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(args.save_ckpt, 'teste_bestVAL.pt'))

            else:
                patience_counter += 1

            # if patience_counter > patience:
            #    break
            
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            if epoch % 10 == 0:
                np.save(os.path.join(args.save_losses, 'teste_train_loss.npy'), all_train_loss)
                np.save(os.path.join(args.save_losses, 'teste_val_loss.npy'), all_val_loss)
            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), os.path.join(args.save_ckpt, f'teste_lastEpoch_{epoch}.pt'))
       
        print(f'--- FINISHED {args.model_name} TRAINING ---')



if __name__ == '__main__':

    args = getArgs('speed')    

    #TRAINING
    trainingBcos(args).training(args)

