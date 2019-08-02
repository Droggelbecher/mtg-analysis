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

    inc_cols = np.random.choice(devotion_cols, len(X))
    Xinc = X.copy()
    Xinc[np.arange(0, len(X)),inc_cols] += 1
    Xinc[cmc_index] += 1

    # dec_cols = np.random.choice(devotion_cols, len(X))
    Xdec = X.copy()
    # Xdec[np.arange(0, len(X)),dec_cols] -= 1
    Xdec[cmc_index] -= 1

    for i in np.arange(len(X)):
        # print(i, len(X))
        if np.all(Xdec[:, devotion_cols][i] == 0):
            Xdec[i, cmc_index] += 1
            continue

        col = np.random.randint(0, len(devotion_cols))
        while Xdec[i, devotion_cols[col]] == 0:
            col = np.random.randint(0, len(devotion_cols))
            # print(col, Xdec[:,devotion_cols][i])
        Xdec[i, devotion_cols[col]] -= 1


    Yinc = np.ones(len(Xinc))
    Ydec = -Yinc

    return (
        np.vstack((X, Xinc, Xdec)),
        np.hstack((Y, Yinc, Ydec))
    )


def get_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(40,)),
      tf.keras.layers.Dense(40, activation=tf.nn.relu),
      tf.keras.layers.Dense(40, activation=tf.nn.relu),
      # tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.Dense(10, activation=tf.nn.softmax)
      tf.keras.layers.Dense(1, activation=tf.nn.tanh)
    ])
    model.compile(optimizer='adam',
                  # loss='sparse_categorical_crossentropy',
                  loss='mse',
                  metrics=['mse', 'mean_absolute_error'])
    return model


def train_model(df, names):
    test_size = 0.1

    df = df.apply(pd.to_numeric)
    all_columns = df.columns.values
    x_columns_gt = sorted([x for x in all_columns if x not in ignore], reverse = True)

    # TODO: This is an ugly crutch, maybe we should do augmentation on the df
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

    # Take test set from the unaugmented data,
    # also store according names

    if True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(len(indices) * (1.0 - test_size))
        train_indices, test_indices = indices[:split], indices[split:]
        # print(train_indices, test_indices)
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test, names_test = X[test_indices], Y[test_indices], np.array(names)[test_indices]

    X_train_a, Y_train_a = augment(X_train, Y_train)

    model = get_model()

    model.load_weights('model.h5')
    # model.fit(X_train_a, Y_train_a, epochs=1000)
    # model.save_weights('model.h5')

    return model, X_test, Y_test, names_test


def print_by_score(df, names):
    model, X_test, Y_test, names_test = train_model(df, names)

    X_test_a, Y_test_a = augment(X_test, Y_test)
    names_a = list(names_test) + [str(n) + '+' for n in names_test] + [str(n) + '-' for n in names_test]

    Y2_a = model.predict(X_test_a)[:, 0]
    losses = (Y2_a - Y_test_a) ** 2
    ind = np.argsort(losses)

    # print("names=", names)
    # print("Y2=", Y2)
    # print("Y_test=", Y_test)
    # print("losses=", losses)
    # print("ind=", ind)

    losses = losses[ind]
    names_a = np.array(names_a)[ind]
    Y2_a = Y2_a[ind]
    Y_test_a = Y_test_a[ind]

    columns = (
        ('Name',  '{:30s}', names_a),
        ('loss',  '{:5.2f}', losses),
        # ('CMC',   '{:4f}', list(df['cmc'])),
        ('pred.', '{:5.2f}', Y2_a),
        ('GT', '{:5.2f}', Y_test_a),
        # ('pwr',   '{:4.0f}', list(df['power'])),
        )

    def print_range(i = None, j = None):
        colnames, fstrings, values = zip(*columns)
        fstr = ' '.join(fstrings)
        # print("colnames=", colnames)
        # print("fstrings=", fstrings)
        # print("values=", values)
        for row in list(zip(*values))[i:j]:
            # print('fstr=',fstr)
            # print('row=',row)
            print(fstr.format(*row))

    print(str(model))
    print("=" * 30 + " ====== ======")
    print("Name" + " " * 26 + " loss   pred.")
    print("=" * 30 + " ====== ======")

    print_range(None, 10)
    print("...")
    i = int(len(names) / 2) - 5
    j = i + 10
    print_range(i, j)
    print("...")
    print_range(-100, None)

    print("=" * 30 + " ====== ======")

np.random.seed(42)

if __name__ == '__main__':
    df, names = jsonutil.read_allsets(sys.argv[1])
    df = preprocess_dataframe(df)
    print('df.shape = {}'.format(df.shape))
    print_by_score(df, names)
    #plot(df, names)
    #analyze_components(df, names)
    # evaluate_model(df, names)
    #print(r)
    #print_ridge_coefs(df, names)
    #print_by_score(df, names)


