#!/usr/bin/env python3

import sys
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import pandas as pd

import jsonutil
from plot import plot_pca_2d, plot_pca_3d

# target columns that we want to learn to predict
targets = [
    'cmc',
    'abs_devotion_red', 'abs_devotion_green', 'abs_devotion_white', 'abs_devotion_blue', 'abs_devotion_black'
    ]

# features that will not give us interesting coefficients
ignore = set([
    'rel_devotion_red', 'rel_devotion_green', 'rel_devotion_white', 'rel_devotion_blue', 'rel_devotion_black',
    'abs_devotion_red', 'abs_devotion_green', 'abs_devotion_white', 'abs_devotion_blue', 'abs_devotion_black',

    # We know already its a creature...
    'is_creature', 'is_land', 'is_planeswalker', 'is_sorcery', 'is_instant', 'has_unblockable', 'loyalty',
    ])

def plot(df, names):
    import matplotlib.pyplot as plt
    df = df.apply(pd.to_numeric)
    df = df.astype(float) # converts bools to floats, needed for radviz

    # Takes forever for this dataset and shows way to much data to see anything
    sys.stderr.write("please stand by while computing scatter matrix...\n")
    pd.tools.plotting.scatter_matrix(df)
    sys.stderr.write("done.\n")

    # Just ends up with 1 (visible) grey dot at each feature
    #pd.tools.plotting.radviz(df, 'cmc')
    plt.show()

def analyze_components(df, names):
    """
    Do some data visualization.
    """
    df = df.apply(pd.to_numeric)
    X = df.values
    X = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X)
    Xtrans = pca.transform(X)
    plot_pca_3d(Xtrans, pca.components_, df.columns.values, pointlabels = names)


def plot_coef_matrix(columns, feature_names, bias, coefs):
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.ma import masked_array
    from pylab import cm

    colormaps = (
        cm.Greys, # CMC
        cm.Reds,
        cm.Greens,
        cm.Greys, # White
        cm.Blues,
        cm.Greys, # Black
        )

    fig, ax = plt.subplots(figsize = (20, 20))
    ax.set_xlim((0, len(bias)))

    print('--')
    print(coefs.T[:3])
    print('--')

    coefs_w_bias = np.vstack((bias, coefs))

    for i, (coef, color) in enumerate(zip(coefs_w_bias.T, colormaps)):
        print('col {}: {}'.format(i, coef))
        pa = ax.imshow([[x] for x in coef], extent = (i, i + 1, 0, len(coef)), interpolation='nearest', cmap = color, origin='upper')
        for j, v in enumerate(coef):
            print('  {}'.format(v))
            plt.text(i + .1, len(coef) - 1 - j + .3, '{:3.2f}'.format(v), fontsize = 8, color = 'black' if v <= .5 else 'white')

    def rename_feature(f):
        if f.startswith('has_'): f = f[4:]
        if f.startswith('is_'): f = f[3:]
        return f.capitalize()

    def rename_target(t):
        if t.startswith('abs_devotion_'):
            t = t[13:]
        return t.capitalize()

    print(np.arange(len(feature_names)), feature_names)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    plt.yticks(np.arange(len(feature_names) + 1) + 0.5, reversed(['(Bias)'] + [rename_feature(f) for f in feature_names]))
    plt.xticks(np.arange(len(columns)) + 0.5, [rename_target(t) for t in columns], rotation = 'vertical')
    plt.subplots_adjust(bottom=0.25)
    #plt.show()
    plt.savefig('coefficients.png')


def print_ridge_coefs(df, names):
    df = df.apply(pd.to_numeric)

    df['date'] /= (365 * 24 * 60 * 60 * 1000000000.0)
    df['date'] = -df['date'] + 48.0

    all_columns = df.columns.values
    #x_columns = sorted([x for x in all_columns if x not in targets and x not in ignore], reverse = True)
    x_columns = [x for x in all_columns if x not in targets and x not in ignore]
    X = df[x_columns].values
    Y = df[targets].values

    reg = Ridge(alpha = 1.0)
    reg.fit(X, Y)
    coef_ = reg.coef_

    plot_coef_matrix(targets, x_columns, reg.intercept_, reg.coef_.T)
    titles = ['Feature           '] + [t + '    ' for t in targets]

    for t in titles: print('=' * len(t) + ' ', end = '')
    print()
    for t in titles: print(t + ' ', end = '')
    print()
    for t in titles: print('=' * len(t) + ' ', end = '')
    print()

    P = 2
    print('{s:{width}s} '.format(s = '(bias)', width = len(titles[0])), end = '')
    for t, v in zip(titles[1:], reg.intercept_):
        print('{v:+{width}.{prec}f} '.format(v = v, width = len(t) - 2, prec = min(P, len(t) - 3)), end = '')
    print()

    for feature, coef in zip(x_columns, coef_.T):
        if feature in ignore: continue
        print('{s:{width}s} '.format(s = feature, width = len(titles[0])), end = '')
        for t, v in zip(titles[1:], coef):
            print('{v:+{width}.{prec}f} '.format(v = v, width = len(t) - 2, prec = min(P, len(t) - 3)), end = '')
        print()

    for t in titles: print('=' * len(t) + ' ', end = '')
    print()


def evaluate_regressors(df, names):
    regressors = [
        make_pipeline(
            StandardScaler(),
            LinearRegression(),
            ),

        make_pipeline(
            StandardScaler(),
            Ridge(alpha = 1.0),
            ),

        make_pipeline(
            StandardScaler(),
            PolynomialFeatures(2, interaction_only = True),
            Ridge(alpha = 1.0)
            ),

        make_pipeline(
            StandardScaler(),
            PolynomialFeatures(3, interaction_only = True),
            Ridge(alpha = 1.0)
            ),

        #make_pipeline(
            #StandardScaler(),
            #PolynomialFeatures(4, interaction_only = True),
            #Ridge(alpha = 1.0)
            #),

        # Takes forever and score is even worse
        #make_pipeline(
            #StandardScaler(),
            #PolynomialFeatures(5, interaction_only = True),
            #Ridge(alpha = 1.0)
            #),

        # No great score either, and not interpretable
        #make_pipeline(
            #StandardScaler(),
            #MLPRegressor(
                #solver = 'lbfgs',
                #hidden_layer_sizes = (50,),
                #random_state = 42
                #)
            #),
        ]

    df = df.apply(pd.to_numeric)
    all_columns = df.columns.values
    x_columns = [x for x in all_columns if x not in targets]
    X = df[x_columns].values
    Y = df[targets].values

    regressor_scores = []
    for regressor in regressors:
        scores = []
        for train_index, test_index in KFold(10).split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            regressor.fit(X_train, Y_train)
            score = regressor.score(X_test, Y_test)
            scores.append(score)
        regressor_scores.append(sum(scores) / len(scores))

    return regressor_scores




if __name__ == '__main__':
    df, names = jsonutil.read_allsets(sys.argv[1])
    #plot(df, names)
    #analyze_components(df, names)
    #r = evaluate_regressors(df, names)
    #print(r)
    print_ridge_coefs(df, names)


