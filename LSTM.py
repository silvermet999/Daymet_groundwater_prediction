import math
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
from torch.optim import RMSprop, Adam
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


class CustomLearningRateScheduler:
    """Linear warmup, optional hold, then cosine decay."""

    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        target_lr=0.001,
        start_lr=0.0,
        hold=0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.hold = hold
        self.global_step = 0

    def lr_warmup_cosine_decay(self, global_step):
        # Warmup: linearly increase from start_lr to target_lr
        if global_step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.target_lr

            return self.start_lr + (
                self.target_lr - self.start_lr
            ) * global_step / self.warmup_steps

        # Hold the target learning rate
        if global_step <= self.warmup_steps + self.hold:
            return self.target_lr

        # Keep the learning rate at target_lr after total_steps
        if global_step >= self.total_steps:
            return self.target_lr

        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps - self.hold

        if decay_steps <= 0:
            return self.target_lr

        progress = (
            global_step - self.warmup_steps - self.hold
        ) / decay_steps

        return 0.5 * self.target_lr * (
            1 + math.cos(math.pi * progress)
        )

    def step(self):
        new_lr = self.lr_warmup_cosine_decay(self.global_step)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print(
            f"  Learning rate = {new_lr:.8f}"
        )

        self.global_step += 1

# Source - https://stackoverflow.com/a/73704579
# Posted by isle_of_gods, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-28, License - CC BY-SA 4.0

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def nse_loss(y_true, y_pred):
  return torch.sum((y_pred - y_true) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)


class LSTM(nn.Module):
  def __init__(self, hidden_size=3, dropout_rate= 0.3):

    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate

    self.lstm = nn.LSTM(input_size=10, hidden_size=self.hidden_size,
                        num_layers=1, bias=True, batch_first=True)
    self.dropout1 = nn.Dropout(p=self.dropout_rate)
    self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=128)
    self.relu1 = nn.ReLU()
    self.dropout2 = nn.Dropout(p=self.dropout_rate)
    self.fc2 = nn.Linear(in_features=128, out_features=256)
    self.relu2 = nn.ReLU()
    self.dropout3 = nn.Dropout(p=self.dropout_rate)
    self.fc3 = nn.Linear(in_features=256, out_features=1)


  def forward(self, x: torch.Tensor):
    # h_n: the final hidden state for each layer.
    _, (h_n, _) = self.lstm(x)

    # perform prediction only at the end of the input sequence
    out = self.relu1(self.fc1(self.dropout1(h_n[-1, :, :])))
    out = self.relu2(self.fc2(self.dropout2(out)))
    pred = self.fc3(self.dropout3(out))
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
HPi_targetlr =  0.001
HPi_epochs = 50
HPi_batchsize = 512

x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
dataset = utils.CustomDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=HPi_batchsize, shuffle=False, num_workers=4, pin_memory=False)

model1 = LSTM().cuda()

early_stopper = EarlyStopper(patience=3, min_delta=10)
optimizer = Adam(model1.parameters(), lr=HPi_targetlr, eps=0.0001)
scheduler = CustomLearningRateScheduler(
    optimizer=optimizer,
    warmup_steps=3,
    total_steps=50,
    target_lr=0.001,
    start_lr=0.0,
    hold=5,
)

for ep in range(HPi_epochs):
    scheduler.step()
    model1.train()
    avg_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(torch.float).cuda()
        y = y.to(torch.float).cuda()

        optimizer.zero_grad()
        lstm = model1(X)
        loss = nn.functional.mse_loss(lstm, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    total_loss = avg_loss / len(train_loader)
    sys.stdout.write('\r epoch %2d loss %.8f' %
                     (ep, total_loss))