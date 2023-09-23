import openai
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
    if api == "openai":
        set_credentials(key, org)
        model_dict = {"model": model}
    elif api == "azure":
        set_azure_credentials(key, org)
        model_dict = {"engine": model}
    else:
        raise ValueError("Invalid API")

    completion = openai.ChatCompletion.create(
        temperature=0.0, messages=messages, **model_dict
    )
    return completion
