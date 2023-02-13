import torch
import torch.nn as nn
import pandas as pd
from load_data import loadData
from Bcos_nets import resNet34
import torch.optim as optim
from utils import plot_losses
import argparse
import os
from tqdm import tqdm
from args_parameters import getArgs
import numpy as np

class trainingBcos:
    def __init__(self, args):
        self.loader = loadData(args)
        self.model = resNet34(args)
        self.create_paths(args.ckpt_path, args.losses_path)

    @staticmethod
    def create_paths(ckpt_path, losses_path):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if not os.path.exists(losses_path):
            os.makedirs(losses_path)


    def training(self, args):
        
        #LOADING DATA
        train_dataloader, val_dataloader = self.loader.getDataloader()

        #TRAINING PARAMETERS
        lr = 3e-4
        patience = 10
        patience_counter = 0
        best_val_loss = 1e9
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        self.model.to(args.device)



        #############################
        #       TRAINING LOOP       #
        #############################

        print(f' STARTING TRAINING FOR {args.model_name} FOR {args.dataset}')

        for epoch in range(args.epochs):
            epoch_val_losses = []
            epoch_train_losses = []

            self.model.train()
            for imgs, labels in tqdm(train_dataloader):
                imgs, labels = imgs.to(args.device), labels.to(args.device)

                output = self.model(imgs)
                optimizer.zero_grad()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_num += 1
                epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    
            self.model.eval()
            for imgs, labels in val_dataloader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)  

                output = self.model(imgs)
                loss = criterion(imgs, labels)
                epoch_val_losses.append(loss.detach().cpu().numpy())

            train_loss, val_loss = np.mean(epoch_train_losses), np.mean(epoch_val_losses)
 
            #Early Stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.gpt_save_ckpt, args.model_name+'_bestVAL.pt'))

            else:
                patience_counter += 1

            # if patience_counter > patience:
            #    break
            
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            if epoch % 10 == 0:
                np.save(os.path.join(args.save_loss, args.model_name+'_train_loss.npy'), all_train_loss)
                np.save(os.path.join(args.save_loss, args.model_name+'_val_loss.npy'), all_val_loss)
                torch.save(model.state_dict(), os.path.join(args.gpt_save_ckpt, f'{args.model_name}_lastEpoch_{epoch}.pt'))

            #SAVING LOSSES PLOT
            if epoch % 20 == 0:
                train_L = np.load(os.path.join(args.save_loss, args.model_name+'_train_loss.npy'))
                val_L = np.load(os.path.join(args.save_loss, args.model_name+'_val_loss.npy'))
                plot_losses(args, train_L, val_L, args.model_name)


       
        print(f'--- FINISHED {args.model_name} TRAINING ---')



if __name__ == '__main__':

    args = getArgs('local')    

    #TRAINING
    trainingBcos(args).training(args)

