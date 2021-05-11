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
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as torch_data
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from data_loader.dataloader import LSTMTSD_Dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
## Step 1: Load Dataset

path = './dataset/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]  ## 파일명 끝이 .csv인 경우

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
# dataloader = DataLoader(df_set, batch_size=1, drop_last=False, shuffle=True, num_workers=0)

RATIO_SPLIT = 0.85
train_dataset, val_dataset = torch.utils.data.random_split(df_set, [int(len(df_set) * RATIO_SPLIT),
                                                                    len(df_set) - int(len(df_set) * RATIO_SPLIT)])
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset) * RATIO_SPLIT),
                                                                        len(val_dataset) - int(
                                                                            len(val_dataset) * RATIO_SPLIT)])

print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))

## Step 2: Make Dataset Iterable
batch_size = 200
n_iters = 10000000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
# num_epochs = int(num_epochs)
# num_epochs = 100
print(int(num_epochs))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)


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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # print("x.size(0).size, h0.size:", x.size(0), h0.size())
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print("1st out, out.size:", out.size())
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # print("2nd out, out.size:", out.size())

        # out.size() --> 100, 10
        return out


## Step 4: Instantiate Model Class

input_dim = 10
hidden_dim = 100
layer_dim = 1
output_dim = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = LSTMModel_v0(input_dim, hidden_dim, layer_dim, output_dim)
model.to(device)
## Step 5: Instantiate Loss Class
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

## Step 6: Instantiate Optimizer Class
learning_rate = 0.00001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

len(list(model.parameters()))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
print("Begin training.")

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

## Step 7: Train Model
# Number of steps to unroll
seq_dim = 1  # torch.Size([20, 10, 1]) --> torch.Size([20, 10, 1])

iter = 0
correct = 0
for epoch in tqdm(range(1, num_epochs + 1)):
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for i, (inputs, labels, _) in enumerate(train_loader):
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_()
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)

        train_loss.backward()
        optimizer.step()
        iter += 1

        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.resize(len(outputs), 1).type(torch.LongTensor)  # torch.Size([20, 1])
        correct += (predicted.to(device) == labels).sum() / batch_size
        train_acc = 100 * correct / labels.size(0)
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        #loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        #accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(f'Epoch {epoch + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}')


        if iter % 10 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            model.train()
            val_epoch_loss = 0
            val_epoch_acc = 0
            val_loss = []
            val_acc = []
            # Iterate through  validation dataset
            for inputs, labels, _ in val_loader:

                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(-1, seq_dim, input_dim)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                predicted = predicted.resize(len(outputs), 1).type(torch.LongTensor)  # torch.Size([20, 1])
                correct += (predicted.to(device) == labels).sum() / batch_size

            val_acc = 100 * correct / total
            # Print Loss
            #print('Iteration: {}. val_loss: {}. val_acc: {}'.format(iter, val_loss.item(), val_acc))

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

    print(f'Epoch {epoch + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')


# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
y_pred_list = []

y_pred_list = []
with torch.no_grad():
    model.eval()
    i=0
    for inputs, y_test, _ in test_loader:
        inputs, y_test = inputs.to(device), y_test.to(device)
        inputs = inputs.view(-1, seq_dim, input_dim)
        outputs = model(inputs)
        test_loss = criterion(outputs, y_test.type(torch.LongTensor))
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        predicted = predicted.resize(len(outputs), 1).type(torch.LongTensor)  # torch.Size([20, 1])
        correct += (predicted == y_test).sum() / batch_size
        y_pred_list.append(predicted.cpu().numpy())
        # print('y_pred {} log_softmax {} y_test {} '.format(y_pred_tags.cpu(), y_test_pred.cpu(), y_test[i]))
        i+=1
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))

sns.heatmap(confusion_matrix_df, annot=True)
print(classification_report(y_test, y_pred_list))



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
