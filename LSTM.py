import sys

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

def create_sliding_windows(X, y, seq_length):
    X_windows = np.array([
        X[i:i + seq_length]
        for i in range(len(X) - seq_length)
    ])

    y_windows = np.array([
        y[i + seq_length]
        for i in range(len(y) - seq_length)
    ])

    return X_windows, y_windows

def make_sequences(data, n_steps_in, n_input):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + n_steps_in
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

def series_to_supervised(data, n_in=1, n_out=1):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  return pd.DataFrame(agg.astype('float32'))


def prepare_data():
  # _, df = main.scale_site()
  # ID_len = list()
  # new_X_test, new_y_test = list(), list()
  # X_test_stat = list()
  # # df = df[2150001]
  # for i in df:
  #     df.reset_index(inplace=True)
  #     df.rename(columns={"index": "DATE"}, inplace=True)
  #     df["DATE"] = pd.to_datetime(df["DATE"])
  #     df.set_index('DATE', inplace=True)
  #
  #     mask_train = df[i]["mask_train"]
  #     mask_val = df[i]["mask_val"]
  #     mask_test = df[i]["mask_test"]
  #
  #     X_train = df[i]["X_train"][mask_train].to_numpy()
  #     X_val = df[i]["X_val"][mask_val].to_numpy()
  #     X_test = df[i]["X_test"][mask_test].to_numpy()
  #
  #     y_train = df[i]["y_train"][mask_train].to_numpy()
  #     y_val = df[i]["y_val"][mask_val].to_numpy()
  #     y_test = df[i]["y_test"][mask_test].to_numpy()
  #
  #     ID_len.append(i, len(X_test)) # 70598, 80, 10
  #     temp_test_ext = np.concatenate([X_val[-52:], X_test])
  #     temp_X, temp_y = make_sequences(temp_test_ext, 52, 10)
  #     new_X_test.append(temp_X); new_y_test.append(temp_y)
  #     X_test_stat.append(np.repeat())

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

  X_wind_train, y_wind_train = create_sliding_windows(X_train, y_train, seq_length=80)
  X_wind_val, y_wind_val = create_sliding_windows(X_val, y_val, seq_length=80)
  X_wind_test, y_wind_test = create_sliding_windows(X_test, y_test, seq_length=80)


  return X_wind_train, X_wind_val, X_wind_test, y_wind_train, y_wind_val, y_wind_test

  #return X_train, X_val, X_test, y_train, y_val, y_test


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


batch_size = 64
lr = 0.01
epochs = 500

# load data
print("loading")
x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()

dataset = utils.CustomDataset(x_train, y_train)
dataset_val = utils.CustomDataset(x_val, y_val)
dataset_test = utils.CustomDataset(x_test, y_test)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)


print("loaded")
model1 = LSTM().cuda()
print("model")


optimizer = RMSprop(model1.parameters(), lr=lr)


def training():
  model1.train()
  avg_loss=0.0
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


# def validation():
#   model1.eval()
#   avg_loss = 0.0
#   with torch.no_grad():
#     for batch_idx, (data, mask) in enumerate(val_loader):
#       data = data.to(torch.float).cuda()
#       mask = mask.to(torch.bool).cuda()
#
#       recon_batch = model1(data)
#       # nseloss = nse_loss(data, recon_batch, mask=mask)
#       loss = robust_recon_loss(data, recon_batch, mask=mask)
#       # loss = 0.1 * nseloss + 0.9 * reconloss
#       avg_loss += loss.item()
#
#   total_loss = avg_loss / len(val_loader)
#   return total_loss


for ep in range(epochs):
  training_loss = training()
  # if ep % 20 == 18:
  #   val_loss = validation()
  #   print("VAL LOSS =====> ", val_loss)
  # torch.save(model1.state_dict(), f"model{training_loss}.pt")