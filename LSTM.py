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

# from svrg import SVRG


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

# def make_windows(data, window_size):
#   windows = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)
#   return windows.transpose(0, 2, 1)
#
#
# def make_window_masks(mask, window_size):
#   return make_windows(mask, window_size)

def train_valid_test_split(data, mask_seq, n_train_valid_rows):
  data_train_valid = data.iloc[:n_train_valid_rows,:]
  mask_train_valid = mask_seq.iloc[:n_train_valid_rows, :]

  data_test = data.iloc[n_train_valid_rows:,:]
  mask_test = mask_seq.iloc[n_train_valid_rows:, :]

  data_valid, data_train = train_test_split(data_train_valid, test_size=0.4, shuffle= False) # the last 60% data in the first 6 years used for training and the first 40% used for validation.
  mask_valid, mask_train = train_test_split(mask_train_valid, test_size=0.4, shuffle=False)

  return data_train.values, mask_train.values, data_valid.values, mask_valid.values, data_test.values, mask_test.values

def prepare_data_with(hours_of_history, hours_to_predict):
  raw = main.features

  # 1. Build the mask from RAW data, before any filling/scaling touches it.
  mask_raw = build_missingness_mask(raw)

  # 2. Fill gaps so the scaler and windowing arithmetic have no NaNs to choke on.
  data_filled = fill_for_scaling(raw)

  # 3. Fit/scale as before, excluding the test period from scaler fitting.
  scaler = MinMaxScaler()
  scaler.fit(data_filled.iloc[:52608, :])
  q_max = np.max(raw.iloc[:52608, 2])  # manual min/max from RAW (not filled) data
  q_min = np.min(raw.iloc[:52608, 2])
  data_scaled = pd.DataFrame(scaler.transform(data_filled), columns=data_filled.columns)

  # 4. Window BOTH the scaled data and the mask with identical shifts,
  #    so mask columns line up 1:1 with data columns (var%d(t-n) etc.)
  data_sequence = series_to_supervised(data_scaled, hours_of_history, hours_to_predict, dropnan=False)
  mask_sequence = series_to_supervised(mask_raw, hours_of_history, hours_to_predict, dropnan=False)

  # Shifting introduces edge NaNs in both (first/last few rows have no
  # t-n or t+n neighbor). Drop those edge rows from both in lockstep —
  # this is a windowing artifact, not a missing-sensor-reading gap, so
  # dropping here is fine and doesn't reintroduce the masking problem.
  valid_rows = ~data_sequence.isna().any(axis=1)
  data_sequence = data_sequence[valid_rows].reset_index(drop=True)
  mask_sequence = mask_sequence[valid_rows].reset_index(drop=True)
  # mask itself might have NaN at the same edge rows (shift artifact);
  # treat any remaining mask NaN as "not observed" to be safe.
  mask_sequence = mask_sequence.fillna(0.0)

  # NOTE: 52608 was an index into the RAW data. After windowing with
  # series_to_supervised + dropping edge rows, row alignment shifts by
  # hours_of_history rows. Adjust the boundary accordingly:
  n_train_valid_rows = 52608 - hours_of_history

  (train_x, train_mask,
   valid_x, valid_mask,
   test_x, test_mask) = train_valid_test_split(data_sequence, mask_sequence, n_train_valid_rows)

  def split_encoder_decoder(arr, mask_arr):
    rainfall = arr[:, 0::3].reshape(-1, hours_of_history + hours_to_predict, 1)
    discharge_all = arr[:, 2::3].reshape(-1, hours_of_history + hours_to_predict, 1)
    x_discharge = discharge_all[:, :hours_of_history, :]
    y = discharge_all[:, hours_of_history:, :]
    x_et = arr[:, 3 * hours_of_history + 1].reshape(-1, 1)

    # same slicing applied to the mask so y_mask lines up with y exactly
    discharge_mask_all = mask_arr[:, 2::3].reshape(-1, hours_of_history + hours_to_predict, 1)
    y_mask = discharge_mask_all[:, hours_of_history:, :]

    return x_et, x_discharge, rainfall, y, y_mask

  train_x_et, train_x_discharge, train_x_rainfall, train_y, train_y_mask = split_encoder_decoder(train_x, train_mask)
  valid_x_et, valid_x_discharge, valid_x_rainfall, valid_y, valid_y_mask = split_encoder_decoder(valid_x, valid_mask)
  test_x_et, test_x_discharge, test_x_rainfall, test_y, test_y_mask = split_encoder_decoder(test_x, test_mask)

  return (
    [train_x_et, train_x_discharge, train_x_rainfall], train_y, train_y_mask,
    [valid_x_et, valid_x_discharge, valid_x_rainfall], valid_y, valid_y_mask,
    [test_x_et, test_x_discharge, test_x_rainfall], test_y, test_y_mask,
    q_max, q_min
  )


# def prepare_data(window_size=2 , n_train_valid_rows=20000,
#                          valid_fraction=0.4, columns=None):
#
#   raw = main.features
#   if columns is not None:
#     raw = raw[columns]
#
#   mask_raw = build_missingness_mask(raw)
#   data_filled = fill_for_scaling(raw)
#
#   scaler = MinMaxScaler()
#   scaler.fit(data_filled.iloc[:n_train_valid_rows, :])
#   data_scaled = pd.DataFrame(scaler.transform(data_filled), columns=data_filled.columns)
#
#   data_train_valid = data_scaled.iloc[:n_train_valid_rows, :].reset_index(drop=True)
#   mask_train_valid = mask_raw.iloc[:n_train_valid_rows, :].reset_index(drop=True)
#   data_test = data_scaled.iloc[n_train_valid_rows:, :].reset_index(drop=True)
#   mask_test = mask_raw.iloc[n_train_valid_rows:, :].reset_index(drop=True)
#
#   n_valid = int(len(data_train_valid) * valid_fraction)
#   data_train = data_train_valid.iloc[:-n_valid, :].reset_index(drop=True)
#   mask_train = mask_train_valid.iloc[:-n_valid, :].reset_index(drop=True)
#   data_valid = data_train_valid.iloc[-n_valid:, :].reset_index(drop=True)
#   mask_valid = mask_train_valid.iloc[-n_valid:, :].reset_index(drop=True)
#
#   # def window_split(data_df, mask_df):
#   #   windows = make_windows(data_df.to_numpy(dtype='float32'), window_size)
#   #   mask_windows = make_window_masks(mask_df.to_numpy(dtype='float32'), window_size)
#   #   return windows, mask_windows
#
#   train_windows = make_windows(data_train.to_numpy(dtype='float32'), window_size)
#   train_mask = make_window_masks(mask_train.to_numpy(dtype='float32'), window_size)
#   valid_windows = make_windows(data_valid.to_numpy(dtype='float32'), window_size)
#   valid_mask = make_window_masks(mask_valid.to_numpy(dtype='float32'), window_size)
  # valid_windows, valid_mask_windows = window_split(data_valid, mask_valid)
  # test_windows, test_mask_windows = window_split(data_test, mask_test)

  # return {
  #   "train": (train_windows, train_mask),
  #   "valid": (valid_windows, valid_mask),
  #   "test": (data_test, mask_test),
  #   "scaler": scaler,
  #   "data_scaled": data_scaled,
  #   "mask_raw": mask_raw,
  # }


# define custome loss function (you can use the simple 'mse' as well)
def nse_loss(y_true, y_pred, mask=None, eps=1e-6):
  if mask is not None:
    y_true = y_true[mask]
    y_pred = y_pred[mask]
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
               window_size,
               n_vars,
               encoder_hidden=128,
               decoder_hidden=256,
               dense_dims=(256, 64),
               dropout_p=0.2):
    super().__init__()

    self.window_size = window_size
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
    _, (h, _) = self.encoder_lstm(x) #.unsqueeze(1))
    h_last = h[-1]

    decoder_input = h_last.unsqueeze(1).repeat(1, self.window_size, 1)

    decoder_out, _ = self.decoder_lstm(decoder_input)

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




# parameters

batch_size = 64
lr = 0.0005
epochs = 500

# load data
print("loading")
results = prepare_data()
x_train, mask_train = results["train"]
x_val, mask_val = results["valid"]
data_scaled = results["data_scaled"].to_numpy()
mask_raw  = results["mask_raw"].to_numpy()
scaler = results["scaler"]
dataset = utils.CustomDataset(x_train, mask_train)
dataset_val = utils.CustomDataset(x_val, mask_val)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
# train_loader = utils.dataset_function(dataset, batch_size=64, train=True)
# test_loader = utils.dataset_function(dataset, batch_size=32, train=False)
print("loaded")
model1 = EDLSTM(window_size=2, n_vars= 296).cuda()
print("model")


optimizer = SVRG(model1.parameters(), lr=lr)

# def cosine_annealing(step, total_steps, lr_max, lr_min):
#   return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
#
#
# sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step,
#                                                                                                  500 * len(
#                                                                                                    train_loader),
#                                                                                                  1,
#                                                                                                  1e-6 / lr))

# sched = ReduceLROnPlateau(optimizer, factor=0.3, patience=15, cooldown=30, min_lr=1e-8)



def training():
  model1.train()
  avg_loss=0.0
  for batch_idx, (data, mask) in enumerate(train_loader):
      data = data.to(torch.float).cuda()
      mask = mask.to(torch.bool).cuda()

      optimizer.zero_grad()
      recon_batch = model1(data)
      # nseloss = nse_loss(data, recon_batch, mask=mask)
      loss = robust_recon_loss(data, recon_batch, mask=mask)
      # loss = 0.1 * nseloss + 0.9 * reconloss
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
      optimizer.step()
      avg_loss += loss.item()

  total_loss = avg_loss / len(train_loader)
  sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %
                       (ep, batch_idx + 1, len(train_loader), total_loss))
  # sched.step(total_loss)


def validation():
  model1.eval()
  avg_loss = 0.0
  with torch.no_grad():
    for batch_idx, (data, mask) in enumerate(val_loader):
      data = data.to(torch.float).cuda()
      mask = mask.to(torch.bool).cuda()

      recon_batch = model1(data)
      # nseloss = nse_loss(data, recon_batch, mask=mask)
      loss = robust_recon_loss(data, recon_batch, mask=mask)
      # loss = 0.1 * nseloss + 0.9 * reconloss
      avg_loss += loss.item()

  total_loss = avg_loss / len(val_loader)
  return total_loss

def stitch_overlapping_windows(window_outputs, n_rows, window_size, n_vars):
  n_windows = window_outputs.shape[0]
  assert n_windows == n_rows - window_size + 1

  sum_recon = np.zeros((n_rows, n_vars), dtype='float32')
  count = np.zeros((n_rows, 1), dtype='float32')

  for i in range(n_windows):
      sum_recon[i:i + window_size] += window_outputs[i]
      count[i:i + window_size] += 1

  # rows at the very start/end are covered by fewer windows; count
  # handles that correctly via the division below
  return sum_recon / count


@torch.no_grad()
def impute_with_model(model_name, window_size=2, batch_size=256):

    model1.load_state_dict(torch.load(model_name))
    model = model1.cuda()
    model.eval()

    n_rows, n_vars = data_scaled.shape

    data_windows = make_windows(data_scaled, window_size)

    n_windows = data_windows.shape[0]
    all_recon = np.zeros_like(data_windows)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = torch.tensor(data_windows[start:end], dtype=torch.float).cuda()
        recon = model(batch)  # (batch, window_size, n_vars)
        all_recon[start:end] = recon.cpu().numpy()

    # Stitch overlapping window reconstructions into one continuous series
    recon_full = stitch_overlapping_windows(all_recon, n_rows, window_size, n_vars)

    # Original mask, aligned row-for-row with data_scaled/recon_full

    # Keep real observed values; substitute reconstruction only where missing
    imputed_scaled = np.where(mask_raw.astype(bool), data_scaled, recon_full)

    imputed_real = scaler.inverse_transform(imputed_scaled)

    imputed_df = pd.DataFrame(imputed_real, columns=None)  # set columns=your_feature_columns
    return imputed_df

# imputed_df = impute_with_model("model0.14691463726758958.pt")
# imputed_df.to_csv('imputed_output.csv', index=False)

for ep in range(epochs):
  training_loss = training()
  if ep % 20 == 18:
    val_loss = validation()
    print("VAL LOSS =====> ", val_loss)
    torch.save(model1.state_dict(), f"model{val_loss}.pt")


