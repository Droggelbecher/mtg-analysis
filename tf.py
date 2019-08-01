#!/usr/bin/env python3

import tensorflow as tf
import jsonutil
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from mtg_analysis import preprocess_dataframe

# target columns that we want to learn to predict
targets = [
    'cmc',
    'abs_devotion_red', 'abs_devotion_green', 'abs_devotion_white', 'abs_devotion_blue', 'abs_devotion_black'
    ]

# features that will not give us interesting coefficients
ignore = set([
    'rel_devotion_red', 'rel_devotion_green', 'rel_devotion_white', 'rel_devotion_blue', 'rel_devotion_black',
    #'abs_devotion_red', 'abs_devotion_green', 'abs_devotion_white', 'abs_devotion_blue', 'abs_devotion_black',

    # We know already its a creature...
    'is_creature', 'is_land', 'is_planeswalker', 'is_sorcery', 'is_instant', 'has_unblockable', 'loyalty',
    ])

def augment(X, Y):

    # Randomly increase/decrease devotion to one color
    devotion_cols = np.array(list(devotion_indices))
        # 'abs_devotion_red',
        # 'abs_devotion_green',
        # 'abs_devotion_white',
        # 'abs_devation_blue',
        # 'abs_devotion_black'
    # ]

    # print("devotion_cols", devotion_cols)
    inc_cols = np.random.choice(devotion_cols, len(X))
    # print("inc_cols=", inc_cols)
    Xinc = X.copy()
    # for i, c in enumerate(inc_cols):
        # print(i, c)
        # Xinc[i][c] += 1
    Xinc[np.arange(0, len(X)),inc_cols] += 1
    Xinc[cmc_index] += 1

    dec_cols = np.random.choice(devotion_cols, len(X))
    Xdec = X.copy()
    Xdec[np.arange(0, len(X)),dec_cols] -= 1
    # for i, c in enumerate(inc_cols):
        # Xdec[i][c] -= 1
    Xdec[cmc_index] -= 1

    Yinc = np.ones(len(Xinc))
    Ydec = -Yinc

    return (
        np.vstack((X, Xinc, Xdec)),
        np.hstack((Y, Yinc, Ydec))
    )


def get_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(40,)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      # tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.Dense(10, activation=tf.nn.softmax)
      tf.keras.layers.Dense(1, activation=tf.nn.tanh)
    ])
    model.compile(optimizer='adam',
                  # loss='sparse_categorical_crossentropy',
                  loss='mse',
                  metrics=['mse', 'mean_absolute_error'])
    return model


def evaluate_model(df, names):
    df = df.apply(pd.to_numeric)
    all_columns = df.columns.values
    x_columns_gt = sorted([x for x in all_columns if x not in ignore], reverse = True)

    global devotion_indices
    global cmc_index

    devotion_indices = set()
    for i, col in enumerate(x_columns_gt):
        if col.startswith('abs_devotion_'):
            devotion_indices.add(i)
        elif col == 'cmc':
            cmc_index = i


    X = df[x_columns_gt].values
    Y = np.zeros(X.shape[0]) # 0 = correct costs

    X, Y = augment(X, Y)

    # Shuffle, so kfold makes sene for a train/test split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = np.array(X[indices], copy=True)
    Y = np.array(Y[indices], copy=True)

    model = get_model()

    # scores = []
    print("X.shape=", X.shape, Y.shape)
    for train_index, test_index in KFold(10).split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train, epochs=20)
        # score = model.evaluate(X_train, Y_train)
        # scores.append(sum(score) / len(score))
        # print(
            # min(scores),
            # sum(scores) / len(scores),
            # max(scores)
        # )

    return model


def print_by_score(df, names):

if __name__ == '__main__':
    df, names = jsonutil.read_allsets(sys.argv[1])
    df = preprocess_dataframe(df)
    print('df.shape = {}'.format(df.shape))
    #plot(df, names)
    #analyze_components(df, names)
    evaluate_model(df, names)
    #print(r)
    #print_ridge_coefs(df, names)
    #print_by_score(df, names)


