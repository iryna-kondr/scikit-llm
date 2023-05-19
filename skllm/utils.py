import numpy as np 
import pandas as pd

def to_numpy(X):
    if isinstance(X, pd.Series):
        X = X.to_numpy()
    elif isinstance(X, list):
        X = np.asarray(X)
    if isinstance(X, np.ndarray):
        X = np.squeeze(X)
    return X