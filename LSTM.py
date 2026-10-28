import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
import main
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

def make_sequences(X, y, seq_length):
    X_windows = np.array([
        X[i:i + seq_length]
        for i in range(len(X) - seq_length)
    ])

    y_windows = np.array([
        y[i + seq_length]
        for i in range(len(y) - seq_length)
    ])

    return X_windows, y_windows

def prepare_data():
    df, _ = main.scale_site()
    # df = df[2150001]

    mask_train = df["mask_train"]
    mask_val = df["mask_val"]
    mask_test = df["mask_test"]

    X_train = df["X_train"][mask_train].to_numpy()
    X_val = df["X_val"][mask_val].to_numpy()
    X_test = df["X_test"][mask_test].to_numpy()

    y_train = df["y_train"][mask_train].to_numpy()
    y_val = df["y_val"][mask_val].to_numpy()
    y_test = df["y_test"][mask_test].to_numpy()

    X_train_wind, y_train_wind = make_sequences(X_train, y_train, 52)

    return X_train_wind, X_val, X_test, y_train_wind, y_val, y_test


def nse_loss(y_true, y_pred):
  return torch.sum((y_pred - y_true) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)


class LSTM(nn.Module):
  def __init__(self, hidden_size=10, dropout_rate= 0.0):
    """Initialize model

    :param hidden_size: Number of hidden units/LSTM cells
    :param dropout_rate: Dropout rate of the last fully connected
        layer. Default 0.0
    """
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate

    # create required layer
    self.lstm = nn.LSTM(input_size=10, hidden_size=self.hidden_size,
                        num_layers=1, bias=True, batch_first=True)
    self.dropout = nn.Dropout(p=self.dropout_rate)
    self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)


  def forward(self, x: torch.Tensor):
    """Forward pass through the Network.

    :param x: Tensor of shape [batch size, seq length, num features]
        containing the input data for the LSTM network.

    :return: Tensor containing the network predictions
    """
    # if x.dim() == 2:
    #   x = x.unsqueeze(1)
    # h_n: the final hidden state for each layer.
    _, (h_n, _) = self.lstm(x)

    # perform prediction only at the end of the input sequence
    pred = self.fc(self.dropout(h_n[-1, :, :]))
    return pred



# identify KGE, NSE for evaluation
def nse(y_true, y_pred):
  return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)

def kge(y_true, y_pred):
  kge_r = np.corrcoef(y_true,y_pred)[1][0]
  kge_a = np.std(y_pred)/np.std(y_true)
  kge_b = np.mean(y_pred)/np.mean(y_true)
  return 1-np.sqrt((kge_r-1)**2+(kge_a-1)**2+(kge_b-1)**2)


HP_seeds = [171, 206, 380, 471, 570, 624, 643, 778, 808, 973]

# Hyperparameters
n_steps_in = 52
HPi_lstm_size = 128
HPi_static_size =  128
HPi_comb_size =  256
HPi_dropout =  0.3
HPi_targetlr =  0.001
HPi_epochs = 50
HPi_batchsize = 512

print("loading")
x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
dataset = utils.CustomDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=HPi_batchsize, shuffle=False, num_workers=4, pin_memory=False)


print("loaded")
model1 = LSTM().cuda()
print("model")


optimizer = RMSprop(model1.parameters(), lr=HPi_targetlr)
# pth_out = "./results/results_s2_cor/"
# for ii in range(len(HP_seeds)):
#     HPi_seed = HP_seeds[ii]
#
#     # create folder for outputs
#     pth_out_i = pth_out + '/run' + str(HPi_seed)
#     os.mkdir(pth_out_i)
#
#     # set seed
#     np.random.seed(HPi_seed)
#     torch.cuda.manual_seed(HPi_seed)
for ep in range(HPi_epochs):
    model1.train()
    avg_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(torch.float).cuda()
        y = y.to(torch.float).cuda()

        optimizer.zero_grad()
        lstm = model1(X)
        loss = nse_loss(y, lstm)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    total_loss = avg_loss / len(train_loader)
    sys.stdout.write('\r epoch %2d %d/%d loss %.8f' %
                     (ep, batch_idx + 1, len(train_loader), total_loss))