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
## Step 1: Load Dataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.tensorboard import SummaryWriter

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())
print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())
print(test_dataset.test_data.size())
print(test_dataset.test_labels.size())

## Step 2: Make Dataset Iterable
batch_size = 200
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


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
        #h0.Size([1, 200, 100]) 200
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())) #out : torch.Size([200, 28, 100])
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])#out : torch.Size([200, 10])

    # out.size() --> 100, 10
        return out


## Step 4: Instantiate Model Class

input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

## Step 5: Instantiate Loss Class
criterion = nn.CrossEntropyLoss()

## Step 6: Instantiate Optimizer Class
learning_rate = 0.1

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
seq_dim = 28

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        print("1st images.size(),labels.size():", images.size(),labels.size())

    # Load images as a torch tensor with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim).requires_grad_()
        print("2nd images.size(),labels.size():", images.size(),labels.size())
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)
        print("outputs, labels:", outputs, labels)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        print("outputs.size(),labels.size():", outputs.size(), labels.size())
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

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Resize images
                images = images.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                print("predicted, outputs:", predicted, outputs)
                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

