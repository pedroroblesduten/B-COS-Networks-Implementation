import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score

def getResults(true, pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = c_matrix.ravel()
    recall = recall_score(true, pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    df = pd.DataFrame()
    df['accuracy'] = accuracy
    df['precision'] = precision
    df['recall'] = recall

    print('-- CONFUSION MATRIX -- ')
    print(c_matrix)
    print('-- CLASSIFICATION METRICS --')
    print(df)

    return accuracy

def getExplanationImage(args, img, grads, smooth=15, alpha_percentile=99.5):
    contrib = (img*grads).sum(0, keepdim=True)
    contrib = contrib[0]
    
    rgb_grad = (grads/(grads.abs().max(0, keepdim=True)[0] + 1e-22))
    rgb_grad = rgb.grad.clamp()

    rgb_grad = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:]+1e-12))

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
    # Only show positive contributions
    alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth-1)//2)
    alpha = to_numpy(alpha)
    alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)

    rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
    # Reshaping to [H, W, C]
    grad_image = rgb_grad.transpose((1, 2, 0))
    grad_image = torch.tensor(grad_image)
    grad_image = grad_image.permute(2, 1, 2)
    return grad_image



class trainingBcos:
    def __init__(self, args):
        self.loader = loadData(args)
        self.model = resNet34(args)
        self.create_paths(args.save_ckpt, args.save_losses)
        self.model = getModels(args)

    @staticmethod
    def create_paths(ckpt_path, losses_path):
        print('*preparing for traing*')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if not os.path.exists(losses_path):
            os.makedirs(losses_path)
            print(ckpt_path, losses_path)

    def getModels(self, args):
        # LOADING MODEL
        transf = resNet34(args).to(args.device)
        if args.gpt_load_ckpt is not None:
            path  = args.gpt_load_ckpt.split('ckpt/')[-1]
            print(f' -> LOADING MODEL: {path}')
            transf.load_state_dict(torch.load(args.load_ckpt, map_location=args.device), strict=False)
        else:
            print(f' -> LOADING MODEL: no checkpoint, intializing randomly')

        return model


    def evaluateMetric(self, args):
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
                    
        self.model.eval()
        for imgs, labels in tqdm(val_dataloader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            true = labels
            labels = F.one_hot(labels, num_classes=10)
            self.model.zero_grad()
            logits = self.model(imgs)

            max_values, max_indices = torch.max(logits, dim=-1)

            # Gather the corresponding class labels
            pred = torch.gather(torch.arange(10), dim=0, index=max_indices.unsqueeze(1)).squeeze(1)

            acc = getResults(true, pred)

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
            imgs, labels = imgs.to(args.device).retain_grad(), labels.to(args.device)
            true = labels
            labels = F.one_hot(labels, num_classes=10)
            self.model.zero_grad()
            output = self.model(imgs)

            loss = criterion(output, labels.float())
            loss.backward()

            probs = F.softmax(logits, dim=-1)
            
            max_values, max_indices = torch.max(probs, dim=-1)

            max_probs, max_img_in_batch = torch.max(max_values, dim=0)

            img_to_explain = imgs[max_img_in_batch, ...]
            explanation = getExplanationImage(img_to_explain, img_to_explain.grad())

            ori_images.append(img_to_explain)
            exp_images.append(exp_images)

        real_fake_images = torch.cat((ori_images[:5], exp_images.add(1).mul(0.5)[:5]))
        vutils.save_image(real_fake_images, os.path.join(args.save_results_path, f'explanation_results.jpg'), nrow=5)
               
        print(f'--- FINISHED {args.model_name} EXPLANATIONS GENERATION ---')



if __name__ == '__main__':

    args = getArgs('local')    

    #TRAINING
    trainingBcos(args).training(args)
