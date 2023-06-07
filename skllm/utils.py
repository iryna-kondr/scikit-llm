import numpy as np
import pandas as pd

from typing import Any

def to_numpy(X: Any) -> np.ndarray:
    """
    Converts a pandas Series or list to a numpy array.

    Parameters
    ----------
    X : Any
        The data to convert to a numpy array.
    
    Returns
    -------
    X : np.ndarray
    """
    if isinstance(X, pd.Series):
        X = X.to_numpy().astype(object)
    elif isinstance(X, list):
        X = np.asarray(X, dtype=object)
    if isinstance(X, np.ndarray) and len(X.shape) > 1:
        X = np.squeeze(X)
    return X


def find_json_in_string(string: str) -> str:
    """
    Finds the JSON object in a string.
    
    Parameters
    ----------
    string : str
        The string to search for a JSON object.
    
    Returns
    -------
    json_string : str
    """

    start = string.find("{")
    end = string.rfind("}")
    if start != -1 and end != -1:
        json_string = string[start : end + 1]
    else:
        json_string = "{}"
    return json_string
