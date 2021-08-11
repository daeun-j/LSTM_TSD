import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import pandas as pd
import numpy as np
import csv
# from models import RNNModel_CUDA
from dataloader_old import *
from utils import *
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()

parser.add_argument('--MERGE', type=int, default=5)
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--fft', type=int, default=4)
parser.add_argument('--stat', type=int, default=1)
parser.add_argument('--layer_dim', type=int, default=1)
# parser.add_argument('--split_ratio', type=float, default=0.9)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--num_gpu', type=int, default=0)
parser.add_argument('--fix_num', type=int, default=725)
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--scheduler', type=str, default="StepLR")

args = parser.parse_args()

output_dim = 3
seq_dim = 1

class RNNModel_CUDA(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel_CUDA, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])

        return out


def multiclass_accuracy(outputs, batch_size):
    _, predicted = torch.max(outputs.data, 1)
    acc = ((predicted == labels)*1).sum()/batch_size *100
    return acc


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_gpu)
print(f'Training configs: {args}')

hyper_params = {"fft": args.fft, "stat" : args.stat, "MERGE" : args.MERGE, "window_size": args.window_size,
                 "lr" : args.lr, "batch_size" : args.batch_size,"hidden_dim": args.hidden_dim,
                "layer_dim": args.layer_dim}

name = "smpl/R_sh{}_M{}_w{}_fn{}_kf{}_ep{}".format(args.scheduler[:2], args.MERGE, args.window_size, args.fix_num, args.k_folds, args.num_epochs)
#file_name = "R_M{}_w{}_fn{}_kf{}_ep{}".format(args.MERGE, args.window_size, args.fix_num, args.k_folds, args.num_epochs)
"""STEP 2: load data"""

df = pd.DataFrame()

# df = pd.read_csv("dataset/Telegram_1hour_7.csv")
# df.insert(2, "label", int(0))
# df_0 = df[["Time", "Length", "label"]].to_numpy()
#
# df = pd.read_csv("dataset/Zoom_1hour_5.csv")
# df.insert(2, "label", int(1))
# df_1 = df[["Time", "Length", "label"]].to_numpy()
#
# df = pd.read_csv("dataset/YouTube_1hour_2.csv")
# df.insert(2, "label", int(2))
# df_2 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("dataset/Telegram_1hour_7_ws10_or5_ms20.csv")
df_0 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("dataset/Zoom_1hour_5_ws10_or5_ms20.csv")
df_1 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("dataset/YouTube_1hour_2_ws10_or5_ms20.csv")
df_2 = df[["Time", "Length", "label"]].to_numpy()

df_set = sampling(w=args.window_size, fix_num=args.fix_num, df_0= df_0, df_1= df_1, df_2= df_2)

if args.MERGE == 0 or args.MERGE == 7 or args.MERGE == 9:
    df_set = Dataset_raw(df_set, MERGE= args.MERGE)

else:
    df_set = Dataset(df_set, window_size= args.window_size,
                     fft_num= args.fft, stat=args.stat, MERGE= args.MERGE)

kfold = KFold(n_splits=args.k_folds, shuffle=True)
result_eval_dict = {"hyper_params": hyper_params}


"""STEP 3: Make data iterable"""
for fold, (train_ids, test_ids) in enumerate(kfold.split(df_set)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset=df_set, batch_size=args.batch_size, drop_last=False, sampler=train_subsampler)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=df_set, batch_size=args.batch_size, drop_last=False, sampler=test_subsampler)

    x, y = next(iter(test_loader))
    input_dim = x.size()[1]

    #CUDA = "cuda:"+str(args.num_gpu)
    #device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNModel_CUDA(input_dim, args.hidden_dim, args.layer_dim, output_dim)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = schedulers(optimizer, args.scheduler)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, min_lr =1e-8)
    #scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(args.num_epochs):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for tr_i, (inputs, labels) in enumerate(train_loader):
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_()
            if args.scheduler != "ReduceLROnPlateau":
                scheduler.step()
                optimizer.zero_grad()
                outputs = model(inputs)
                if args.MERGE == 0 or args.MERGE == 7 or args.MERGE == 9:
                    labels = labels.view(-1, )
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                train_acc = multiclass_accuracy(outputs, labels.size(0)) #cross entropy
                optimizer.step()
            # ReduceLRONPlateau : scheduler.step(train_loss)
            else: #args.scheduler == "ReduceLROnPlateau"
                optimizer.zero_grad()
                outputs = model(inputs)
                if args.MERGE == 0 or args.MERGE == 7 or args.MERGE == 9:
                    labels = labels.view(-1, )
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                train_acc = multiclass_accuracy(outputs, labels.size(0)) #cross entropy
                optimizer.step()
                scheduler.step(train_loss)
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            if tr_i % 300 == 0:
                print(f'Train Epoch {tr_i + 0:05}: | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f}')
                torch.save({'epoch': tr_i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "loss": train_loss}, "./Weights/"+name+".pt")

        y_pred_list, y_test_list = [], []
        test_name = "{}kf_e{}".format(fold, epoch)
        test_outputs_sets = np.array([])

        with torch.no_grad():
            model.eval()
            start = datetime.now()
            for inputs, y_test in test_loader:
                inputs, y_test = inputs.to(device), y_test.to(device)
                inputs = inputs.view(-1, seq_dim, input_dim)

                y_test_pred = model(inputs)
                y_pred_softmax = torch.log_softmax(y_test_pred, 1)
                _, y_pred_tags = torch.max(y_pred_softmax, 1)

                y_pred_list.append(y_pred_tags.cpu().numpy())
                y_test_list.append(y_test.cpu().numpy())
            end = datetime.now()

            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
            y_test_list = [a.squeeze().tolist() for a in y_test_list]

            if type(y_pred_list[-1]) == list:
                y_pred_list_fo = [j for sub in y_pred_list for j in sub]
                y_test_list_fo = [j for sub in y_test_list for j in sub]

            else:
                y_pred_list_fo = [j for sub in y_pred_list[:-1] for j in sub]
                y_test_list_fo = [j for sub in y_test_list[:-1] for j in sub]

                y_pred_list_fo.append(y_pred_list[-1])
                y_test_list_fo.append(y_test_list[-1])

            #result_test_file = "result/"+name+"/"+test_name

            test_dict = validate(np.array(y_test_list_fo, dtype = int) , np.array(y_pred_list_fo, dtype = int))#, result_test_file)

            test_time = "{}".format(end-start)
            print("test_time: ", test_time)
            test_dict[str(fold)+"_"+str(epoch)+"time"] = test_time
            result_test_dict = {test_name: test_dict}
            result_eval_dict.update(result_test_dict)

result_eval_dict_name = "result/"+name
with open(result_eval_dict_name+'.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(result_eval_dict.keys())
    w.writerow(result_eval_dict.values())