
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd

import jsonutil
from plot import plot_pca_2d, plot_pca_3d


def analyze_components(df, names):
    df = df.apply(pd.to_numeric)
    X = df.values
    X = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X)
    Xtrans = pca.transform(X)
    plot_pca_2d(Xtrans, pca.components_, df.columns.values, pointlabels = names)

def linear_regression(df, names):
    df = df.apply(pd.to_numeric)
    #print(df['date'])

    targets = [
        'cmc',
        'abs_devotion_red',
        'abs_devotion_green',
        'abs_devotion_white',
        'abs_devotion_blue',
        'abs_devotion_black'
        ]

    ignore = [
        'rel_devotion_red',
        'rel_devotion_green',
        'rel_devotion_white',
        'rel_devotion_blue',
        'rel_devotion_black',
        'abs_devotion_red',
        'abs_devotion_green',
        'abs_devotion_white',
        'abs_devotion_blue',
        'abs_devotion_black'
        ]


    all_columns = df.columns.values
    x_columns = [x for x in all_columns if x not in targets]

    X = df[x_columns].values
    Y = df[targets].values

    #scaler_x = StandardScaler()
    #scaler_y = StandardScaler()
    #X = scaler_x.fit_transform(X)
    #Y = scaler_y.fit_transform(Y)

    # TODO: test set for scoring to figure out whether this actually makes sense!
    #reg = LinearRegression(normalize = False)
    reg = Ridge(alpha = 0.1)
    reg.fit(X, Y)

    # coef is #targets x #x_columns

    coef_ = reg.coef_
    #coef_ = scaler_x.inverse_transform(reg.coef_)

    for colname, coef in zip(x_columns, coef_.T):
        if colname in ignore: continue
        print('{}:'.format(colname))
        for t, v in zip(targets, coef):
            print('  {}: {:6.4f}'.format(t, v))

    print(reg.intercept_)

    # TRAINING set score
    print(reg.score(X, Y))


if __name__ == '__main__':
    df, names = jsonutil.read_allsets(sys.argv[1])
    #analyze_components(df, names)
    linear_regression(df, names)


