import json
from time import sleep

import openai

from skllm.openai.credentials import set_credentials
from skllm.utils import find_json_in_string


def construct_message(role, content):
    if role not in ("system", "user", "assistant"):
        raise ValueError("Invalid role")
    return {"role": role, "content": content}


def get_chat_completion(messages, key, org, model="gpt-3.5-turbo", max_retries=3):
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


def extract_json_key(json_, key):
    original_json = json_
    for i in range(2):
        try:
            json_ = original_json.replace("\n", "")
            if i == 1:
                json_ = json_.replace("'", "\"")
            json_ = find_json_in_string(json_)
            as_json = json.loads(json_)
            if key not in as_json.keys():
                raise KeyError("The required key was not found")
            return as_json[key]
        except Exception:
            if i == 0:
                continue
            return None