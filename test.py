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
import numpy as np
import torch.nn as nn
import torch.utils.data as torch_data
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from data_loader.dataloader import LSTMTSD_Dataset
import pandas
## Step 1: Load Dataset

path = './dataset/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')] ## 파일명 끝이 .csv인 경우

## csv 파일들을 DataFrame으로 불러와서 concat

df = pd.DataFrame()
ylabel = 0
for i in file_list_py:
    df = pd.read_csv(path + i)
    df = df[["Time", "Length"]].to_numpy()
    df_set = LSTMTSD_Dataset(df, ylabel=ylabel, window_size=10, horizon=1, normalize_method="z_score")

    if ylabel == 0:
        concate_dataset = df_set
    else:
        concate_dataset = torch.utils.data.ConcatDataset([concate_dataset, df_set])
    ylabel += 1



RATIO_SPLIT = 0.8
train_dataset, val_dataset = torch.utils.data.random_split(concate_dataset, [int(len(concate_dataset)*RATIO_SPLIT),len(concate_dataset) - int(len(concate_dataset)*RATIO_SPLIT)])
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset)*RATIO_SPLIT),len(val_dataset) - int(len(val_dataset)*RATIO_SPLIT)])



print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))


## Step 2: Make Dataset Iterable
batch_size = 10
n_iters = 10000
num_epochs = n_iters / (len(train_dataset) / batch_size)
#num_epochs = int(num_epochs)
num_epochs = 300

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)




## Step 3: Create Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


## Step 4: Instantiate Model Class

input_dim = 20
hidden_dim = 100
layer_dim = 1
output_dim = 3

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

## Step 5: Instantiate Loss Class
criterion = nn.MSELoss()

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
seq_dim = 1

iter = 0
for epoch in range(num_epochs):
    for i, (inputs, labels, meta) in enumerate(train_loader):
        # Load images as a torch tensor with gradient accumulation abilities
        inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(inputs)

    # Calculate Loss: softmax --> mse
        loss = criterion(inputs, labels)
        # Getting gradients w.r.t. parameters
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

        if iter % 100 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for inputs, labels, _ in test_loader:
                # Resize images
                inputs = inputs.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(inputs)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

