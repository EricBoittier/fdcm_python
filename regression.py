import pickle
import pandas as pd
import sklearn
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import os

def scale_min_max(data, x):
    return (x - data.min()) / (data.max() - data.min())


def inv_scale_min_max(x, dmin, dmax):
    return x * (dmax - dmin) + dmin


def scale_Z(data, x):
    return (x - data.mean()) / (data.std())


def inv_scale_Z(x, dmean, dstd):
    return x * dstd + dmean


def process_data(df):
    lcs_df = pd.DataFrame(dict(df["lcs"])).T
    rename = {i: f"q{i // 3}ax{i % 3}" for i in range(30)}
    lcs_df.rename(columns=rename, inplace=True)
    uptri_df = pd.DataFrame(dict(df["uptriDM"])).T
    return lcs_df, uptri_df


def run_regression(df_name, alpha=1e-5, test_size=0.5):
    RANDSTATE = 0

    df = pd.read_pickle(df_name)
    lcs_df, uptri_df = process_data(df)

    X = uptri_df


    models = []
    scale_parms = []
    r2 = []
    keys = lcs_df.keys()

    # a kernel for each axis of each charge
    for key in keys:

        y = lcs_df[key]

        y = scale_min_max(y, y)

        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=RANDSTATE)

        kernel = RBF()
        model = KernelRidge(
            alpha=alpha,
            kernel=kernel,
            # random_state=RANDSTATE
        ).fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        r2_train = sklearn.metrics.r2_score(y_train, train_predictions)
        r2_test = sklearn.metrics.r2_score(y_test, test_predictions)

        models.append(model)
        scale_parms.append((lcs_df[key].min(), lcs_df[key].max()))

        r2.append(r2_test)

    output = {"keys": keys,
              "df": df,
              "models": models,
              "scale_parms": scale_parms,
              "r2": r2,
              "test": X_test,
              "train": X_train}

    filehandler = open(f"data/models/{os.path.basename(df_name)}", "wb")
    pickle.dump(output, filehandler)

    return output


