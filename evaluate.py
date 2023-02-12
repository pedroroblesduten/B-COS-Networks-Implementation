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

class evaluateModel:
    def __init__(self, args):
        self.loader = loadData(args)
        self.model = resNet34(args)
        self.create_paths(args.evaluate_path)

    @staticmethod
    def create_paths(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def evaluateClassification(self, args):
        train_dataloader, val_dataloader = self.loader.getDataloader()

        self.model.eval()
        for imgs, labels in val_dataloader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)  

            output = self.model(imgs)

        getResults(labels, output)

        





