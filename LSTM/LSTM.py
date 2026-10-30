import gc
import math
import os
import sys
from datetime import datetime

import pandas as pd
from scipy.stats import stats

import main
import tensorflow as tf
from keras.callbacks import EarlyStopping

import utils

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
    _, df = main.scale_site()
    # df = df[2150001]
    sequences = {}
    for i in df:
        mask_train = df[i]["mask_train"]
        mask_val = df[i]["mask_val"]
        mask_test = df[i]["mask_test"]

        X_train = df[i]["X_train"][mask_train].to_numpy()
        X_val = df[i]["X_val"][mask_val].to_numpy()
        X_test = df[i]["X_test"][mask_test].to_numpy()

        y_train = df[i]["y_train"][mask_train].to_numpy()
        y_val = df[i]["y_val"][mask_val].to_numpy()
        y_test = df[i]["y_test"][mask_test].to_numpy()

        X_train_wind, y_train_wind = make_sequences(X_train, y_train, 80)
        X_val_wind, y_val_wind = make_sequences(X_val, y_val, 80)
        X_test_wind, y_test_wind = make_sequences(X_test, y_test, 80)

        sequences[i] = {
            "X_train": X_train_wind,
            "X_val": X_val_wind,
            "X_test": X_test_wind,

            "y_train": y_train_wind,
            "y_val": y_val_wind,
            "y_test": y_test_wind
        }

    return sequences, df




HP_seeds = [171, 206, 380, 471, 570, 624, 643, 778, 808, 973]

# Hyperparameters
n_steps_in = 80
HPi_lstm_size = 128
HPi_static_size = 128
HPi_comb_size = 256
HPi_dropout = 0.3
HPi_targetlr = 0.001
HPi_epochs = 20
HPi_batchsize = 512

sequences, df = prepare_data()

# ii=0
for ii in range(len(HP_seeds)):

    HPi_seed = HP_seeds[ii]

    # take time
    now1 = datetime.now()

    # set seed
    np.random.seed(HPi_seed)
    tf.random.set_seed(HPi_seed)

    # Callbacks
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    model_mc = tf.keras.callbacks.ModelCheckpoint(filepath="results" + '/model.keras',
                                                  save_best_only=True)

    warmup_steps = 3  # Number of warmup steps
    total_steps = 20  # Total training steps
    target_lr = 0.001  # Peak learning rate
    lr_callback = utils.CustomLearningRateScheduler(warmup_steps, total_steps, target_lr=target_lr)


    for i in df:
        X_train = sequences[i]["X_train"]
        X_val = sequences[i]["X_val"]
        X_test = sequences[i]["X_test"]
        y_train = sequences[i]["y_train"]
        y_val = sequences[i]["y_val"]
        y_test = sequences[i]["y_test"]
        # Input layers for dynamic and static model strands
        model_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))

        # Dynamic model strand
        model = tf.keras.layers.LSTM(HPi_lstm_size)(model_in)
        model = tf.keras.layers.Dropout(HPi_dropout)(model)

        model = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model)
        model = tf.keras.layers.Dropout(HPi_dropout)(model)

        # Define output layer for predictions
        model_output = tf.keras.layers.Dense(units=1, activation='linear', dtype=tf.float32)(model)

        model = tf.keras.Model(inputs=model_in, outputs=model_output)

        # Compile model with appropriate loss function and optimizer
        optimizer = tf.keras.optimizers.Adam(epsilon=0.0001)
        model.compile(loss='mse', optimizer=optimizer)

        # Use fit with generators
        model_history = model.fit(X_train, y_train,
                                  validation_data=(X_val, y_val),
                                  epochs=HPi_epochs, verbose=1,
                                  callbacks=[model_es, model_mc, lr_callback,
                                             utils.BatchProgressCallback(),
                                             utils.ClearMemory()])

        # take time
        now2 = datetime.now()
        timetaken = round((now2 - now1).total_seconds()) / 60
        print('\n timetaken = ' + str(timetaken) + '\n')

        #
        # # predict - with saved model checkpoint
        gc.collect()
        tf.keras.backend.clear_session()
        loaded_model = tf.keras.models.load_model("results" + '/model.keras')
        sim_n = loaded_model.predict(X_test, batch_size=64)
