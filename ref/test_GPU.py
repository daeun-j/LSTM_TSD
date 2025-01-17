## Library
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

from dataloader import Dataset
from models import LSTM_v0_CUDA
from utils import validate

"""STEP 1: Parameters"""
split_ratio = 0.9
batch_size = 2000
n_iters = 10000000
input_dim = 20
hidden_dim = 100
layer_dim = 1
output_dim = 3
seq_dim = 1
learning_rate = 0.00001
window_size=50
fft_num =3
stat = 1
MERGE = 6

"""STEP 2: load data"""
df = pd.DataFrame()

df = pd.read_csv("../dataset/Telegram_1hour_7.csv")
df.insert(2, "label", int(0))
df_0 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("../dataset/Zoom_1hour_5.csv")
df.insert(2, "label", int(1))
df_1 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("../dataset/YouTube_1hour_2.csv")
df.insert(2, "label", int(2))
df_2 = df[["Time", "Length", "label"]].to_numpy()

df_set = np.vstack((df_0, df_1, df_2))
#df_set = Dataset(df_set, window_size=window_size, horizon=1, normalize_method="z_score")


df_set = Dataset(df_set, window_size=window_size,
                      fft_num= fft_num, stat=stat, MERGE= MERGE)



train_dataset, val_dataset = torch.utils.data.random_split(
                df_set, [int(len(df_set) * split_ratio),
                len(df_set) - int(len(df_set) * split_ratio)])

val_dataset, test_dataset = torch.utils.data.random_split(
                val_dataset, [int(len(val_dataset) * split_ratio),
                len(val_dataset) - int(len(val_dataset) * split_ratio)])

print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))

"""STEP 3: Make data iterable"""
# num_epochs = int(num_epochs)
# num_epochs = 100
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
num_epochs = 100
print("num_epochs:", int(num_epochs))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)


x, y = next(iter(test_loader))
input_dim = x.size()[1]

print('type:', type(train_loader), '\n')
first_batch = train_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))
# 총 데이터의 개수는 len(train_loader) *  len(first_batch[0])이다.

"""STEP: 3 Model generation"""
"""STEP 4: Instantiate Model"""
#######################
#  USE GPU FOR MODEL  #
#######################
model = LSTM_v0_CUDA(input_dim, hidden_dim, layer_dim, output_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

## Step 5: Instantiate Loss Class
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

## Step 6: Instantiate Optimizer Class
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
num_epochs = 1

for epoch in range(num_epochs):
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for tr_i, (inputs, labels) in enumerate(train_loader):
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_()

        optimizer.zero_grad()
        outputs = model(inputs)

        train_loss = criterion(outputs, labels)
        train_loss.backward()
        train_acc = multiclass_accuracy(outputs, labels.size(0))
        # Updating parameters
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        #print(f'Train Epoch {tr_i + 0:05}: | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f}')
        if tr_i % 300 == 0:
            print(f'Train Epoch {tr_i + 0:05}: | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f}')
            torch.save({'epoch': tr_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "loss": train_loss}, "./Weights/v0_CUDA_0512.pt")

            """
            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])
            """

    val_outputs_sets = torch.tensor([]).to(device)
    for inputs, labels in val_loader:
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, seq_dim, input_dim)
        outputs = model(inputs)

        val_loss = criterion(outputs, labels)
        #_, predicted = torch.max(outputs.data, 1)
        outputs = outputs.data.max(1)[1]
        val_outputs_pairs = torch.vstack((labels, outputs))
        val_outputs_sets = torch.hstack((val_outputs_pairs, val_outputs_sets))
        #print(val_outputs_sets)

    result_fileresult_file = "result"
    validate(val_outputs_sets[0].to("cpu").numpy(), val_outputs_sets[0].to("cpu").numpy(), window_size, result_fileresult_file)

#STEP 8: TEST

y_pred_list, y_test_list = [], []
with torch.no_grad():
    #model.eval()
    i=0
    for inputs, y_test in test_loader:
        inputs, y_test = inputs.to(device), y_test.to(device)
        inputs = inputs.view(-1, seq_dim, input_dim)
        y_test_pred = model(inputs)
        y_pred_softmax = torch.log_softmax(y_test_pred, 1)
        _, y_pred_tags = torch.max(y_pred_softmax, 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
        y_test_list.append(y_test.cpu().numpy())
        #print('y_pred {} log_softmax {} y_test {} '.format(y_pred_tags.cpu(), y_test_pred.cpu(), y_test[i]))
        i+=1
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test_list = [a.squeeze().tolist() for a in y_test_list]
    #print(y_pred_list, y_test_list)

y_pred_list= [j for sub in y_pred_list for j in sub]
y_test_list= [j for sub in y_test_list for j in sub]
y_test_list= list(map(round, y_test_list))
print(confusion_matrix(y_pred_list,  y_test_list, labels=[0, 1, 2]))
print(classification_report(y_test_list, y_pred_list))


result_fileresult_file = "results"
validate(outputs.to("cpu").numpy(), labels.to("cpu").numpy(), result_fileresult_file)

