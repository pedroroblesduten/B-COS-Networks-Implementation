import torch
import torch.nn as nn
import pandas as pd
from datasets.load_data import loadData
from Bcos_nets import resNet34
import torch.optim as optim

class trainingBcos:
    def __init__(self, args):
        self.loader = loadData(args)
        self.model = resNet34

    @staticmethod
    def prepare_training_mri_vqvae():
        os.makedirs('results_mri_vqvae', exist_ok=True)
        os.makedirs('checkpoints_mri_vqvae', exist_ok=True)

    def training(self):
        
        #LOADING DATA
        train_dataloader, val_dataloader = self.loader.getDataloader(args)

        #TRAINING PARAMETERS
        lr = 3e-4
        patience = 10
        patience_counter = 0
        best_val_loss = 1e9
        criterion = loss
        optimizer = optim.Adam(self.model.parameters())



        #############################
        #       TRAINING LOOP       #
        #############################

        print(f' STARTING TRAINING FOR {args.model_name} FOR {args.datasets}')

        for epoch in range(args.epochs):
            epoch_val_losses = []
            epoch_train_losses = []

            model.train()
            for batch in tqdm(train_dataloader):
                                   
                output = self.model(batch)
                optimizer.zero_grad()
                loss = criterion(X, Y)
                loss.backward()
                optimizer.step()
                iter_num += 1
                epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    
            model.eval()
            for val_batch in X_val_index:
                X, Y = get_inputs(val_batch, args.pkeep)
                X, Y = X.to(args.device), Y.to(args.device)
                logits, loss = model(X, Y)
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
       
        print(f'--- FINISHED {args.model_name} TRAINING ---')a







