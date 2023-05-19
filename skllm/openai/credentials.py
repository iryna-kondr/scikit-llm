import openai

def set_credentials(key: str, org: str):
    openai.api_key = key
    openai.organization = org
