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


x_columns_gt = sorted([x for x in jsonutil.Card._fields if x not in ignore], reverse = True)
print("---- x_columns_gt ----")
for c in x_columns_gt:
    print(c)
print("---- /x_columns_gt ----")


def augment(df, Y, names):
    devotion_cols = [
        'abs_devotion_red', 'abs_devotion_green', 'abs_devotion_white', 'abs_devotion_blue', 'abs_devotion_black'
    ]

    cmc_col = df.columns.get_loc('cmc')

    df_inc = df.copy()
    df_inc.iloc[:, cmc_col] += 1
    inc_cols = np.random.choice(devotion_cols, len(df_inc))
    inc_cols_i = [df.columns.get_loc(c) for c in inc_cols]
    df_inc.iloc[np.arange(0, len(df_inc)), inc_cols_i] += 1
    names_inc = []
    for i, (c,n) in enumerate(zip(inc_cols, names)):
        names_inc.append(n + '+' + c[len('abs_devotion_'):])
    Yinc = np.array(Y + 1., copy=True)

    assert len(df_inc) == len(Yinc) == len(names_inc)

    df_dec = df.copy()
    names_dec = []
    df_dec.iloc[:, cmc_col] -= 1
    Ydec = np.array(Y - 1., copy=True)
    for i, n in enumerate(names):
        row = df_dec.iloc[i]
        if np.all(row[devotion_cols] == 0):
            names_dec.append(n + '-')
            df_dec.iloc[i, cmc_col] += 1
            Ydec[i] += 1
            continue

        col = np.random.randint(0, len(devotion_cols))
        c = devotion_cols[col]
        ci = df_dec.columns.get_loc(c)
        while row[ci] == 0:
            col = np.random.randint(0, len(devotion_cols))
            c = devotion_cols[col]
            ci = df_dec.columns.get_loc(c)
        df_dec.iloc[i, ci] -= 1

        names_dec.append(n + '-' + c[len('abs_devotion_'):])
    
    assert len(df_dec) == len(Ydec) == len(names_dec)

    return (
        pd.concat([df, df_inc, df_dec]),
        np.hstack((Y, Yinc, Ydec)),
        list(names) + names_inc + names_dec
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


def tt_split_n(n, test_size=.1):
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(len(indices) * (1.0 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    return train_indices, test_indices

def tt_split_and_augment(df, names, test_size=.1):
    train_idx, test_idx = tt_split_n(len(df), test_size=test_size)
    Y = np.zeros(len(df))

    # split

    df_train = df.loc[train_idx]
    Y_train = Y[train_idx]
    names_train = np.array(names)[train_idx]

    df_test = df.loc[test_idx]
    Y_test = Y[test_idx]
    names_test = np.array(names)[test_idx]

    # augment

    df_train_a, Y_train_a, names_train_a = augment(df_train, Y_train, names_train)
    X_train_a = df_to_np(df_train_a)

    df_test_a, Y_test_a, names_test_a = augment(df_test, Y_test, names_test)
    X_test_a = df_to_np(df_test_a)

    return (
        X_train_a, Y_train_a, names_train_a,
        X_test_a, Y_test_a, names_test_a,
    )


def df_to_np(df):
    dfnp = df.apply(pd.to_numeric)
    X = dfnp[x_columns_gt].values
    return X


def train_model(X_train_a, Y_train_a, load_existing=True):

    model = get_model()

    if load_existing:
        model.load_weights('model.h5')
    else:
        model.fit(X_train_a, Y_train_a, epochs=100)
        model.save_weights('model.h5')


    model.summary()

    return model


def dump(df):
    cols = [
        'abs_devotion_red',
        'abs_devotion_blue',
        'abs_devotion_black',
        'abs_devotion_white',
        'abs_devotion_green',
        'cmc',
        'has_flying',
        'date',
        'power',
        'toughness',
    ]
    print(df[df['has_flying'] == 1.][cols])

def query_oracle(model):
    base_ptype = { k: 0. for k in x_columns_gt }

    def row(ptype, k, v):
        d = base_ptype.copy()
        d.update(ptype)
        d[k] = v
        return d

    def find_best(prop, from_, to, ptype):
        df = pd.DataFrame([ row(ptype, prop, i) for i in range(from_, to) ], columns=x_columns_gt)
        X = df_to_np(df)
        Y = model.predict(X).reshape(-1)
        i = np.argmin(np.abs(Y))
        return from_ + i, Y[i]

    for f in (0, 1):
        print(f"flying: {f}")
        print("------------------")
        print()

        for power in range(0, 5):
            print(f"pwr={power}  ", end='')
            for toughness in range(1, 5):
                cmc, y = find_best(
                    'cmc', 0, 10, {
                        'has_flying': f,
                        'power': power,
                        'toughness': toughness,
                        'is_common': 1,
                        'date': 0.224658,
                    }
                )
                print(f'{cmc:1d}({y:+3.2f}) ', end='')
            print()


def print_by_score(model, X_test_a, Y_test_a, names_test_a):

    print('Evaluation:', model.evaluate(X_test_a, Y_test_a))

    Y2_a = model.predict(X_test_a)[:, 0]
    losses = (Y2_a - Y_test_a) ** 2
    ind = np.argsort(losses)

    losses = losses[ind]
    names_a = np.array(names_test_a)[ind]
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
        for row in list(zip(*values))[i:j]:
            print(fstr.format(*row))

    print(str(model))
    print("=" * 30 + " ===== ===== ====")
    print("Name" + " " * 26 + " loss  pred. GT")
    print("=" * 30 + " ===== ===== ====")

    # print_range(None, 10)
    # print("...")
    # i = int(len(names) / 2) - 5
    # j = i + 10
    # print_range(i, j)
    # print("...")
    # print_range(-100, None)

    print_range(None, None)

    print("=" * 30 + " ===== ===== ====")

np.random.seed(42)

if __name__ == '__main__':
    df, names = jsonutil.read_allsets(sys.argv[1])
    df = preprocess_dataframe(df)
    print('df.shape = {}'.format(df.shape))

    # dump(df)

    (
        X_train_a, Y_train_a, names_train_a,
        X_test_a, Y_test_a, names_test_a,
    ) = tt_split_and_augment(df, names)

    model = train_model(X_train_a, Y_train_a, load_existing=True)

    print("---- Test scores -----")
    print_by_score(model, X_test_a, Y_test_a, names_test_a)

    print("---- Training scores -----")
    print_by_score(model, X_train_a, Y_train_a, names_train_a)

    query_oracle(model)



