import openai
import json

def construct_message(role, content):
    if role not in ("system", "user", "assistant"):
        raise ValueError("Invalid role")
    return {"role": role, "content": content}

def get_chat_completion(messages, key, org, model="gpt-3.5-turbo", max_retries = 3):
    openai.api_key = key
    openai.organization = org
    for _ in range(max_retries):
        try:
            completion = openai.ChatCompletion.create(
                model=model, temperature=0., messages=messages
            )
            return completion
        except Exception:
            continue

def extract_json_key(json_, key):
    try: 
        as_json = json.loads(json_.replace('\n', '')) 
        if key not in as_json.keys():
            raise KeyError("The required key was not found")
        return as_json[key]
    except Exception as e:
        return None