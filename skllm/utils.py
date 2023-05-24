import numpy as np 
import pandas as pd

def to_numpy(X):
    if isinstance(X, pd.Series):
        X = X.to_numpy().astype(object)
    elif isinstance(X, list):
        X = np.asarray(X, dtype = object)
    if isinstance(X, np.ndarray) and len(X.shape) > 1:
        X = np.squeeze(X)
    return X