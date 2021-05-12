## Library
import device as device
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

from data_loader.dataloader import LSTMTSD_Dataset
from models import LSTM_v0

"""STEP 1: Parameters"""
RATIO_SPLIT = 0.9
batch_size = 200
n_iters = 10000000
input_dim = 20
hidden_dim = 100
layer_dim = 1
output_dim = 3
seq_dim = 1
learning_rate = 0.00001
window_size=5
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

del df

df_set = np.vstack((df_0, df_1, df_2))
df_set = LSTMTSD_Dataset(df_set, window_size=window_size, horizon=1, normalize_method="z_score")

train_dataset, val_dataset = torch.utils.data.random_split(df_set, [int(len(df_set) * RATIO_SPLIT),
                                                                    len(df_set) - int(len(df_set) * RATIO_SPLIT)])

val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset) * RATIO_SPLIT),
                                                                        len(val_dataset) - int(
                                                                            len(val_dataset) * RATIO_SPLIT)])

print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))

"""STEP 3: Make data iterable"""
# num_epochs = int(num_epochs)
# num_epochs = 100
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
print("num_epochs:", int(num_epochs))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)


print('type:', type(train_loader), '\n')

first_batch = train_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))
# 총 데이터의 개수는 len(train_loader) *  len(first_batch[0])이다.

"""STEP: 3 Model generation"""
model = LSTM_v0(input_dim, hidden_dim, layer_dim, output_dim)
model.to(device)
"""STEP 4: Instantiate Model"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#
## Step 5: Instantiate Loss Class
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

## Step 6: Instantiate Optimizer Class


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
len(list(model.parameters()))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

print(model)
def multiclass_accuracy(outputs, batch_size):
    _, predicted = torch.max(outputs.data, 1)
    acc = ((predicted == labels)*1).sum()/batch_size *100
    return acc

"""STEP 7: Training"""
accuracy_stats = {'train': [],"val": []}
loss_stats = {'train': [],"val": []}
val_i = 0
num_epochs = 1000
for epoch in range(num_epochs):
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for tr_i, (inputs, labels, _) in enumerate(train_loader):
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_()

        optimizer.zero_grad()
        outputs = model(inputs)

        train_loss = criterion(outputs, labels.type(torch.LongTensor))
        train_loss.backward()
        train_acc = multiclass_accuracy(outputs, labels.size(0))
        # Updating parameters
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

        if tr_i % 300 == 0:
            print(f'Train Epoch {tr_i + 0:05}: | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f}')
            torch.save(model, 'LSTMModel_v0.pt')

            """
            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])
            """


        # validation start
        val_i += 1
        if tr_i % 500 == 0:
            # Calculate Accuracy
            val_epoch_loss = 0
            val_epoch_acc = 0
            correct = 0
            total = 0
            for inputs, labels, _ in val_loader:
                inputs = inputs.view(-1, seq_dim, input_dim)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.type(torch.LongTensor))
                _, predicted = torch.max(outputs.data, 1)
                val_acc = ((predicted == labels)*1).sum()/batch_size *100
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
            val_acc_total = 100 * correct / total
            print(f'Validation Epoch {val_i + 0:03}: | Validation Acc: {val_acc_total:.3f}')
            torch.save(model, 'LSTMModel_v0.pt')

    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader))
    print(f'Epoch {epoch + 0:05}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

"""STEP 8: TEST"""
# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


y_pred_list, y_test_list = [], []
with torch.no_grad():
    model.eval()
    i=0
    for inputs, y_test, _ in test_loader:
        inputs = inputs.view(-1, seq_dim, input_dim)
        y_test_pred = model(inputs)
        y_pred_softmax = torch.log_softmax(y_test_pred, 1)
        _, y_pred_tags = torch.max(y_pred_softmax, 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
        y_test_list.append(y_test.cpu().numpy())
        print('y_pred {} log_softmax {} y_test {} '.format(y_pred_tags.cpu(), y_test_pred.cpu(), y_test[i]))
        i+=1
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test_list = [a.squeeze().tolist() for a in y_test_list]
    print(y_pred_list, y_test_list)

from itertools import chain
y_pred_list= [j for sub in y_pred_list for j in sub]
y_test_list= [j for sub in y_test_list for j in sub]
print(confusion_matrix(y_pred_list,  y_test_list, labels=[0, 1, 2]))
print(classification_report(y_test_list, y_pred_list))