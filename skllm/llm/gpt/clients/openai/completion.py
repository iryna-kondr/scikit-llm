import openai
from openai import OpenAI
from skllm.llm.gpt.clients.openai.credentials import (
    set_azure_credentials,
    set_credentials,
)
from skllm.utils import retry


@retry(max_retries=3)
def get_chat_completion(
    messages: dict,
    key: str,
    org: str,
    model: str = "gpt-3.5-turbo",
    api="openai",
    json_response=False,
):
    """Gets a chat completion from the OpenAI API.

    Parameters
    ----------
    messages : dict
        input messages to use.
    key : str
        The OPEN AI key to use.
    org : str
        The OPEN AI organization ID to use.
    model : str, optional
        The OPEN AI model to use. Defaults to "gpt-3.5-turbo".
    max_retries : int, optional
        The maximum number of retries to use. Defaults to 3.
    api : str
        The API to use. Must be one of "openai" or "azure". Defaults to "openai".

    Returns
    -------
    completion : dict
    """
    if api in ("openai", "custom_url"):
        client = set_credentials(key, org)
    elif api == "azure":
        client = set_azure_credentials(key, org)
    else:
        raise ValueError("Invalid API")
    model_dict = {"model": model}
    if json_response and api == "openai":
        model_dict["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(
        temperature=0.0, messages=messages, **model_dict
    )
    return completion
