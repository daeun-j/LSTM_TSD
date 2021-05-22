import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from dataloader import Dataset
from utils import validate
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
parser = argparse.ArgumentParser()

parser.add_argument('--window_size', type=int, default=30)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--fft', type=int, default=3)
parser.add_argument('--stat', type=int, default=1)
parser.add_argument('--MERGE', type=int, default=0)
parser.add_argument('--layer_dim', type=int, default=1)
parser.add_argument('--split_ratio', type=float, default=0.9)
parser.add_argument('--n_iters', type=int, default=100000)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=2)



output_dim = 3
seq_dim = 1


args = parser.parse_args()
print(f'Training configs: {args}')
name = "epochs{}_merge{}_w{}_lr{}_f{}".format(args.num_epochs,
                                              args.MERGE, args.window_size, args.lr, args.fft)
name_merge = "merge{}".format(args.MERGE)
hyper_params = {"fft": args.fft, "stat" : args.stat, "MERGE" : args.MERGE, "window_size": args.window_size,
                "lr" : args.lr, "batch_size" : args.batch_size,"epoch": args.epoch, "hidden_dim": args.hidden_dim,
                "n_iters": args.n_iters, "split_ratio": args.split_ratio, "layer_dim": args.layer_dim}

"""STEP 2: load data"""

df = pd.DataFrame()

df = pd.read_csv("./dataset/Telegram_1hour_7.csv")
df.insert(2, "label", int(0))
df_0 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("./dataset/Zoom_1hour_5.csv")
df.insert(2, "label", int(1))
df_1 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("./dataset/YouTube_1hour_2.csv")
df.insert(2, "label", int(2))
df_2 = df[["Time", "Length", "label"]].to_numpy()

df_set = np.vstack((df_0, df_1, df_2))

df_set = Dataset(df_set, window_size= args.window_size,
                 fft_num= args.fft, stat=args.stat, MERGE= args.MERGE)

train_dataset, val_dataset = torch.utils.data.random_split(
    df_set, [int(len(df_set) * args.split_ratio),
             len(df_set) - int(len(df_set) * args.split_ratio)])

val_dataset, test_dataset = torch.utils.data.random_split(
    val_dataset, [int(len(val_dataset) * args.split_ratio),
                  len(val_dataset) - int(len(val_dataset) * args.split_ratio)])

print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))

"""STEP 3: Make data iterable"""
# num_epochs = int(num_epochs)
# num_epochs = 100
#num_epochs = int(args.n_iters / (len(train_dataset) / args.batch_size))
print("num_epochs:", int(args.num_epochs))

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)


x, y = next(iter(test_loader))
input_dim = x.size()[1]

"""RF"""


"""Ensemble"""



"""Naive Bayes"""