"""
Step 1: Load Dataset
Step 2: Make Dataset Iterable
Step 3: Create Model Class
Step 4: Instantiate Model Class
Step 5: Instantiate Loss Class
Step 6: Instantiate Optimizer Class
Step 7: Train Model
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as torch_data
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from data_loader.dataloader import LSTMTSD_Dataset
import numpy as np
## Step 1: Load Dataset

path = './dataset/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')] ## 파일명 끝이 .csv인 경우

## csv 파일들을 DataFrame으로 불러와서 concat

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
df_set = LSTMTSD_Dataset(df_set, window_size=5, horizon=1, normalize_method="z_score")
#dataloader = DataLoader(df_set, batch_size=1, drop_last=False, shuffle=True, num_workers=0)

RATIO_SPLIT = 0.8
train_dataset, val_dataset = torch.utils.data.random_split(df_set, [int(len(df_set)*RATIO_SPLIT),len(df_set) - int(len(df_set)*RATIO_SPLIT)])
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset)*RATIO_SPLIT),len(val_dataset) - int(len(val_dataset)*RATIO_SPLIT)])

print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))


## Step 2: Make Dataset Iterable
batch_size = 200
n_iters = 10000000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
#num_epochs = int(num_epochs)
#num_epochs = 100
print(int(num_epochs))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)


## Step 3: Create Model Class
class LSTMModel_v0(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel_v0, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        #print("x.size(0).size, h0.size:", x.size(0), h0.size())
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #print("1st out, out.size:", out.size())
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        #print("2nd out, out.size:", out.size())

    # out.size() --> 100, 10
        return out

## Step 4: Instantiate Model Class

input_dim = 10
hidden_dim = 100
layer_dim = 1
output_dim = 3

#model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = LSTMModel_v0(input_dim, hidden_dim, layer_dim, output_dim)

## Step 5: Instantiate Loss Class
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

## Step 6: Instantiate Optimizer Class
learning_rate = 0.00001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

len(list(model.parameters()))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

## Step 7: Train Model
# Number of steps to unroll
seq_dim = 1 #torch.Size([20, 10, 1]) --> torch.Size([20, 10, 1])

iter = 0
for epoch in range(num_epochs):
    for i, (inputs, labels, _) in enumerate(train_loader):
        #print("1st inputs.size(),labels.size():", inputs.size(), labels.size())
        #torch.Size([20, 10, 1]) --> torch.Size([200, 1, 1])
        inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_()
        #inputs = inputs.view(seq_dim, input_dim).requires_grad_()
        #print("2nd inputs.size(),labels.size():", inputs.size(), labels.size())
        optimizer.zero_grad()
        # print("inputs: ", inputs.size())
        # print("labels: ", labels.size())
        outputs = model(inputs) #output: torch.Size([20, 1])
        # print("outputs, labels:", outputs, labels)
        # print("outputs.size(),labels.size():", outputs.size(), labels.size())
    # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels.type(torch.LongTensor))


        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        # ...학습 중 손실(running loss)을 기록하고
        # writer.add_scalar('training loss',
        #                   loss / 1000,
        #                   epoch * len(train_loader) + i)

        # ...무작위 미니배치(mini-batch)에 대한 모델의 예측 결과를 보여주도록
        # Matplotlib Figure를 기록합니다
        # writer.add_figure('predictions vs. actuals',
        #                   plot_classes_preds(model, images, labels),
        #                   global_step=epoch * len(train_loader) + i)
        if iter % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through  validation dataset
            for inputs, labels, _ in val_loader:
                # Resize images
                inputs = inputs.view(-1, seq_dim, input_dim)
                # Forward pass only to get logits/output
                outputs = model(inputs) #torch.Size([20, 1])
                # print("outputs:", outputs)
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1) #torch.Size([20])
                # print("predicted, predicted.size():", predicted, predicted.size())
                # print("outputs.data, outputs.data.size():", outputs.data, outputs.data.size())
            # Total number of labels
                total += labels.size(0)
                #print("total:", total)
                # Total correct predictions
                predicted = predicted.resize(len(outputs), 1).type(torch.LongTensor) #torch.Size([20, 1])
                #print("predicted, outputs:", predicted.size(), outputs.size())
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

