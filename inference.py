import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from load_data import loadData
from Bcos_nets import resNet34
import torch.optim as optim
from utils import plot_losses
import argparse
import os
from tqdm import tqdm
from args_parameters import getArgs
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score
from torchvision import utils as vutils

def getResults(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    #tn, fp, fn, tp = c_matrix
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    dict_metrics = {'accuuracy' : [accuracy],
                    'precision' : [precision],
                    'recall' : [recall]}

    df = pd.DataFrame.from_dict(dict_metrics)

    print('-- CONFUSION MATRIX -- ')
    print(c_matrix)
    print('-- CLASSIFICATION METRICS --')
    print(df)

    return accuracy

def getExplanationImage(img, grads, smooth=15, alpha_percentile=99.5):
    contrib = (img*grads).sum(0, keepdim=True)
    contrib = contrib[0]
    
    
    rgb_grad = (grads / (grads.abs().max(0, keepdim=True)[0] + 1e-12))
    rgb_grad = rgb_grad.clamp(0).cpu().numpy()
    print('rgb_grad', rgb_grad.shape)
    
    #rgb_grad = (rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:]+1e-12)).cpu().numpy()
        
    alpha = (grads.norm(p=2, dim=0, keepdim=True))
    print('a', alpha.shape)
    # Only show positive contributions
    alpha = torch.where(contrib[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth-1)//2)
    alpha = (alpha).cpu().numpy()
    alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)
    print('b', alpha.shape)

    rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
    print('c', rgb_grad.shape)
    # Reshaping to [H, W, C]
    grad_image = rgb_grad.transpose((1, 2, 0))
    grad_image = torch.tensor(grad_image)
    grad_image = grad_image.permute(2, 0, 1)
    print('aaaaa', grad_image.shape)
    
    return grad_image



class inferenceBcos:
    def __init__(self, args):
        self.loader = loadData(args)
        self.model = resNet34(args)
        self.model = self.getModels(args)

    def getModels(self, args):
        # LOADING MODEL
        model = resNet34(args).to(args.device)
        if args.load_ckpt is not None:
            path  = args.load_ckpt.split('ckpt/')[-1]
            print(f' -> LOADING MODEL: {path}')
            model.load_state_dict(torch.load(args.load_ckpt, map_location=args.device), strict=False)
        else:
            print(f' -> LOADING MODEL: no checkpoint, intializing randomly')

        return model


    def evaluateMetrics(self, args):
        print('======================================')
        print('|                                    |')
        print('|      WELCOME TO BCOS INFERENCE     |')
        print('|                                    |')
        print('======================================')
        
        #LOADING DATA
        train_dataloader, val_dataloader = self.loader.getDataloader()

        #TRAINING PARAMETERS
        criterion = nn.BCEWithLogitsLoss()

        
        #############################
        #       INFERENCE LOOP      #
        #############################

        print(f' STARTING CLASSIFICATION WITH {args.model_name} FOR {args.dataset}')                 
        pred_all = []
        true_all = []
        self.model.eval()
        for imgs, labels in tqdm(val_dataloader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            true = labels
            labels = F.one_hot(labels, num_classes=10)
            self.model.zero_grad()
            logits = self.model(imgs)
            probs = F.softmax(logits, dim=-1)

            max_values, pred = torch.max(probs, dim=-1)
            #print(max_indices[:10])
            pred, true = list(pred.cpu().detach().numpy()), list(true.cpu().detach().numpy())
            pred_all = np.concatenate((pred_all, pred))
            true_all = np.concatenate((true_all, true))

        

            # Gather the corresponding class labels
            #pred = torch.gather(torch.arange(10, device=args.device), dim=0, index=max_indices.unsqueeze(1)).squeeze(1)
        pred_all = torch.tensor(pred_all) #.view(-1)
        true_all = torch.tensor(true_all) # .view(-1)
        acc = getResults(true_all, pred_all)

    def getExplanations(self, args):

        #LOADING DATA
        train_dataloader, val_dataloader = self.loader.getDataloader()

        #TRAINING PARAMETERS
        criterion = nn.BCEWithLogitsLoss()

        
        #############################
        #       INFERENCE LOOP      #
        #############################

        print(f' STARTING EXPLANATIONS GENERATION WITH {args.model_name} FOR {args.dataset}')


        ori_images = []
        exp_images = []
        self.model.eval()
        for imgs, labels in tqdm(val_dataloader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            imgs.requires_grad = True
            true = labels
            labels = F.one_hot(labels, num_classes=10)
            self.model.zero_grad()
            logits = self.model(imgs)

            loss = criterion(logits, labels.float())
            loss.backward()

            probs = F.softmax(logits, dim=-1)
            
            max_values, max_indices = torch.max(probs, dim=-1)

            max_probs, max_img_in_batch = torch.max(max_values, dim=0)

            explanation = getExplanationImage(imgs[max_img_in_batch], imgs.grad[max_img_in_batch])
            
            ori_images.append(imgs[max_img_in_batch])

            exp_images.append(explanation)
        
        ori_images = torch.stack(ori_images).cpu()
        exp_images = torch.stack(exp_images).cpu()
        print(ori_images.shape)
        print(exp_images.shape)
        real_fake_images = torch.cat((ori_images[:5], exp_images.add(1).mul(0.5)[:5]))
        vutils.save_image(real_fake_images, os.path.join(args.save_results_path, f'explanation_results.jpg'), nrow=5)
               
        print(f'--- FINISHED {args.model_name} EXPLANATIONS GENERATION ---')



if __name__ == '__main__':

    args = getArgs('speed')    

    #TRAINING
    inferenceBcos(args).getExplanations(args)
