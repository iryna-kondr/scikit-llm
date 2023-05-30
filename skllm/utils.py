import numpy as np
import pandas as pd


def to_numpy(X):
    if isinstance(X, pd.Series):
        X = X.to_numpy().astype(object)
    elif isinstance(X, list):
        X = np.asarray(X, dtype=object)
    if isinstance(X, np.ndarray) and len(X.shape) > 1:
        X = np.squeeze(X)
    return X


def find_json_in_string(string):
    start = string.find("{")
    end = string.rfind("}")
    if start != -1 and end != -1:
        json_string = string[start : end + 1]
    else:
        json_string = {}
    return json_string
