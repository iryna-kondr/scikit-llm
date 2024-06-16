import json
from typing import Any
import numpy as np
import pandas as pd
from functools import wraps
from time import sleep
import re

def to_numpy(X: Any) -> np.ndarray:
    """Converts a pandas Series or list to a numpy array.

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
        # do not squeeze the first dim
        X = np.squeeze(X, axis=tuple([i for i in range(1, len(X.shape))]))
    return X

# TODO: replace with re version below
def find_json_in_string(string: str) -> str:
    """Finds the JSON object in a string.

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



def re_naive_json_extractor(json_string: str, expected_output: str = "object") -> str:
    """Finds the first JSON-like object or array in a string using regex.
    
    Parameters
    ----------
    string : str
        The string to search for a JSON object or array.

    Returns
    -------
    json_string : str
        A JSON string if found, otherwise an empty JSON object.
    """
    json_pattern = json_pattern = r'(\{.*\}|\[.*\])'
    match = re.search(json_pattern, json_string, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return r"{}" if expected_output == "object" else "[]"




def extract_json_key(json_: str, key: str):
    """Extracts JSON key from a string.

    json_ : str
        The JSON string to extract the key from.
    key : str
        The key to extract.
    """
    original_json = json_
    for i in range(2):
        try:
            json_ = original_json.replace("\n", "")
            if i == 1:
                json_ = json_.replace("'", '"')
            json_ = find_json_in_string(json_)
            as_json = json.loads(json_)
            if key not in as_json.keys():
                raise KeyError("The required key was not found")
            return as_json[key]
        except Exception:
            if i == 0:
                continue
            return None


def retry(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    error_type = type(e).__name__
                    sleep(2**attempt)
            err_msg = (
                f"Could not complete the operation after {max_retries} retries:"
                f" `{error_type} :: {error_msg}`"
            )
            print(err_msg)
            raise RuntimeError(err_msg)

        return wrapper

    return decorator
