import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
import main
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau



def get_series_for_outlier_detection(data_scaled, mask_raw, column_name):
  series = data_scaled[column_name].to_numpy()
  mask = mask_raw[column_name].to_numpy()
  return series, mask


def mad_bounds(series, mask=None, threshold=3.5):
  s = series[mask.astype(bool)] if mask is not None else series
  median = np.median(s)
  MAD = np.median(np.abs(s - median))
  if MAD == 0:
    MAD = 1e-9
  modified_z = 0.6745 * (series - median) / MAD
  lower = median - (threshold / 0.6745) * MAD
  upper = median + (threshold / 0.6745) * MAD
  is_outlier = np.abs(modified_z) > threshold
  return is_outlier, lower, upper

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


def prepare_data(): #(hours_of_history, hours_to_predict):
  df, _ = main.scale_site()

  mask_train = df["mask_train"]
  mask_val = df["mask_val"]
  mask_test = df["mask_test"]

  X_train = df["X_train"][mask_train].to_numpy()
  X_val = df["X_val"][mask_val].to_numpy()
  X_test = df["X_test"][mask_test].to_numpy()

  y_train = df["y_train"][mask_train].to_numpy()
  y_val = df["y_val"][mask_val].to_numpy()
  y_test = df["y_test"][mask_test].to_numpy()

  return X_train, X_val, X_test, y_train, y_val, y_test

  # data_sequence = series_to_supervised(X_train, hours_of_history, hours_to_predict, dropnan=False)
  # mask_sequence = series_to_supervised(mask_train, hours_of_history, hours_to_predict, dropnan=False)

  # train_x_rainfall = X_train[:, 0::3].reshape(-1, hours_of_history + hours_to_predict, 1)
  # train_discharge = X_train[:, 2::3].reshape(-1, hours_of_history + hours_to_predict, 1)
  # train_x_discharge = train_discharge[:, :hours_of_history, :]
  # train_y = train_discharge[:, hours_of_history:, :]
  # train_x_et = X_train[:, 3 * hours_of_history + 1].reshape(-1, 1)  # the current hour et
  #
  # valid_x_rainfall = data_valid[:, 0::3].reshape(-1, hours_of_history + hours_to_predict, 1)
  # valid_discharge = data_valid[:, 2::3].reshape(-1, hours_of_history + hours_to_predict, 1)
  # valid_x_discharge = valid_discharge[:, :hours_of_history, :]
  # valid_y = valid_discharge[:, hours_of_history:, :]
  # valid_x_et = data_valid[:, 3 * hours_of_history + 1].reshape(-1, 1)  # the current hour et
  #
  # test_x_rainfall = data_test[:, 0::3].reshape(-1, hours_of_history + hours_to_predict, 1)
  # test_discharge = data_test[:, 2::3].reshape(-1, hours_of_history + hours_to_predict, 1)
  # test_x_discharge = test_discharge[:, :hours_of_history, :]
  # test_y = test_discharge[:, hours_of_history:, :]
  # test_x_et = data_test[:, 3 * hours_of_history + 1].reshape(-1, 1)  # the current hour et
  #
  # return [train_x_et, train_x_discharge, train_x_rainfall], train_y, [valid_x_et, valid_x_discharge,
  #                                                                     valid_x_rainfall], valid_y, [test_x_et,
  #                                                                                                  test_x_discharge,
  #                                                                                                  test_x_rainfall], test_y, q_max, q_min
  # )


# define custome loss function (you can use the simple 'mse' as well)
def nse_loss(y_true, y_pred, eps=1e-6):
  denom = torch.sum((y_true - torch.mean(y_true)) ** 2)
  denom = torch.clamp(denom, min=eps)
  return torch.sum((y_pred - y_true) ** 2) / denom

def robust_recon_loss(y_true, y_pred, mask=None, delta=1.0):
  if mask is not None:
    y_true = y_true[mask]
    y_pred = y_pred[mask]
  return nn.functional.huber_loss(y_pred, y_true, delta=delta)

class EDLSTM(nn.Module):
  def __init__(self,
               n_vars,
               encoder_hidden=128,
               decoder_hidden=256,
               dense_dims=(256, 64),
               dropout_p=0.2):
    super().__init__()
    self.n_vars = n_vars

    self.encoder_lstm = nn.LSTM(input_size=n_vars, hidden_size=encoder_hidden, batch_first=True)
    self.decoder_lstm = nn.LSTM(input_size=encoder_hidden, hidden_size=decoder_hidden, batch_first=True)

    dense_layers = []
    in_dim = decoder_hidden
    for dim in dense_dims:
      dense_layers.append(nn.Linear(in_dim, dim))
      dense_layers.append(nn.ReLU())
      dense_layers.append(nn.Dropout(dropout_p))
      in_dim = dim
    self.dense_stack = nn.Sequential(*dense_layers)

    self.output_layer = nn.Linear(in_dim, n_vars)
    self.output_activation = nn.Sigmoid()


  def forward(self, x):
    _, (h, _) = self.encoder_lstm(x)
    # h_last = h[-1]
    # print("t", h_last.shape)
    #
    # decoder_input = h_last.unsqueeze(1)
    # print("fo", decoder_input.shape)

    decoder_out, _ = self.decoder_lstm(h)

    x = self.dense_stack(decoder_out)
    x = self.output_activation(self.output_layer(x))
    return x #.squeeze(1)



# identify KGE, NSE for evaluation
def nse(y_true, y_pred):
  return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)

def kge(y_true, y_pred):
  kge_r = np.corrcoef(y_true,y_pred)[1][0]
  kge_a = np.std(y_pred)/np.std(y_true)
  kge_b = np.mean(y_pred)/np.mean(y_true)
  return 1-np.sqrt((kge_r-1)**2+(kge_a-1)**2+(kge_b-1)**2)


batch_size = 64
lr = 0.0005
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
model1 = EDLSTM(n_vars= 32).cuda()
print("model")


optimizer = utils.SVRG(model1.parameters(), lr=lr)


def training():
  model1.train()
  avg_loss=0.0
  for batch_idx, (X, y) in enumerate(train_loader):
      X = X.to(torch.float).cuda()
      y = y.to(torch.bool).cuda()

      optimizer.zero_grad()
      recon_batch = model1(X)
      loss = nse_loss(X, recon_batch)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
      optimizer.step()
      avg_loss += loss.item()

  total_loss = avg_loss / len(train_loader)
  sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %
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
#
# def stitch_overlapping_windows(window_outputs, n_rows, window_size, n_vars):
#   n_windows = window_outputs.shape[0]
#   assert n_windows == n_rows - window_size + 1
#
#   sum_recon = np.zeros((n_rows, n_vars), dtype='float32')
#   count = np.zeros((n_rows, 1), dtype='float32')
#
#   for i in range(n_windows):
#       sum_recon[i:i + window_size] += window_outputs[i]
#       count[i:i + window_size] += 1
#
#   # rows at the very start/end are covered by fewer windows; count
#   # handles that correctly via the division below
#   return sum_recon / count
#
#
# @torch.no_grad()
# def impute_with_model(model_name, window_size=2, batch_size=256):
#
#     model1.load_state_dict(torch.load(model_name))
#     model = model1.cuda()
#     model.eval()
#
#     n_rows, n_vars = data_scaled.shape
#
#     data_windows = make_windows(data_scaled, window_size)
#
#     n_windows = data_windows.shape[0]
#     all_recon = np.zeros_like(data_windows)
#
#     for start in range(0, n_windows, batch_size):
#         end = min(start + batch_size, n_windows)
#         batch = torch.tensor(data_windows[start:end], dtype=torch.float).cuda()
#         recon = model(batch)  # (batch, window_size, n_vars)
#         all_recon[start:end] = recon.cpu().numpy()
#
#     # Stitch overlapping window reconstructions into one continuous series
#     recon_full = stitch_overlapping_windows(all_recon, n_rows, window_size, n_vars)
#
#     # Original mask, aligned row-for-row with data_scaled/recon_full
#
#     # Keep real observed values; substitute reconstruction only where missing
#     imputed_scaled = np.where(mask_raw.astype(bool), data_scaled, recon_full)
#
#     imputed_real = scaler.inverse_transform(imputed_scaled)
#
#     imputed_df = pd.DataFrame(imputed_real, columns=None)  # set columns=your_feature_columns
#     return imputed_df

# imputed_df = impute_with_model("model0.14691463726758958.pt")
# imputed_df.to_csv('imputed_output.csv', index=False)

for ep in range(epochs):
  training_loss = training()
  # if ep % 20 == 18:
  #   val_loss = validation()
  #   print("VAL LOSS =====> ", val_loss)
  torch.save(model1.state_dict(), f"model{training_loss}.pt")