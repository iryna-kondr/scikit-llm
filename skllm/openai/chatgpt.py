import json
from time import sleep
from typing import Any

import openai

from skllm.openai.credentials import set_credentials
from skllm.utils import find_json_in_string


def construct_message(role: str, content: str) -> dict:
    """
    Construct a message for the OpenAI API.
    
    Parameters
    ----------
    role : str
        The role of the message. Must be one of "system", "user", or "assistant".
    content : str
        The content of the message.

    Returns
    -------
    message : dict
    """
    if role not in ("system", "user", "assistant"):
        raise ValueError("Invalid role")
    return {"role": role, "content": content}


def get_chat_completion(messages: str, key: str, org: str, model: str="gpt-3.5-turbo", max_retries: int=3):
    """
    Get a chat completion from the OpenAI API.
    
    Parameters
    ----------
    messages : str
        input messages to use.
    key : str
        The OPEN AI key to use.
    org : str
        The OPEN AI organization to use.
    model : str, optional
        The OPEN AI model to use. Defaults to "gpt-3.5-turbo".
    max_retries : int, optional
        The maximum number of retries to use. Defaults to 3.
    
    Returns
    -------
    completion : dict
    """
    set_credentials(key, org)
    error_msg = None
    error_type = None
    for _ in range(max_retries):
        try:
            completion = openai.ChatCompletion.create(
                model=model, temperature=0.0, messages=messages
            )
            return completion
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            sleep(3)
    print(
        f"Could not obtain the completion after {max_retries} retries: `{error_type} ::"
        f" {error_msg}`"
    )


def extract_json_key(json_: str, key: str) -> Any:
    """
    Extracts a key from a JSON string.
    
    Parameters
    ----------
    json_ : str
        The JSON string to extract the key from.
    key : str
        The key to extract.
    """
    try:
        json_ = json_.replace("\n", "")
        json_ = find_json_in_string(json_)
        as_json = json.loads(json_)
        if key not in as_json.keys():
            raise KeyError("The required key was not found")
        return as_json[key]
    except Exception:
        return None
