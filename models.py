import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM_v0(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_v0, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


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

class LSTM_v0_CUDA(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_v0_CUDA, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class, L1, L2, L3):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature,L1)
        self.layer_2 = nn.Linear(L1, L2)
        self.layer_3 = nn.Linear(L2, L3)
        self.layer_out = nn.Linear(L3, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(L1)
        self.batchnorm2 = nn.BatchNorm1d(L2)
        self.batchnorm3 = nn.BatchNorm1d(L3)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class MulticlassClassification_CUDA(nn.Module):
    def __init__(self, num_feature, num_class, L1, L2, L3):
        super(MulticlassClassification_CUDA, self).__init__()

        self.layer_1 = nn.Linear(num_feature,L1)
        self.layer_2 = nn.Linear(L1, L2)
        self.layer_3 = nn.Linear(L2, L3)
        self.layer_out = nn.Linear(L3, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(L1)
        self.batchnorm2 = nn.BatchNorm1d(L2)
        self.batchnorm3 = nn.BatchNorm1d(L3)

    def forward(self, x):
        #x.view(-1, input_dim)
        x = self.layer_1(x)
        #x = self.batchnorm1()
        x = self.relu(x)

        x = self.layer_2(x)
        #x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        #x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


