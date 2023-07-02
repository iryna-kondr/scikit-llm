from time import sleep

import openai

from skllm.openai.credentials import set_azure_credentials, set_credentials


def construct_message(role: str, content: str) -> dict:
    """Constructs a message for the OpenAI API.

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


def get_chat_completion(
    messages: dict,
    key: str,
    org: str,
    model: str = "gpt-3.5-turbo",
    max_retries: int = 3,
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
        set_azure_credentials(key)
        model_dict = {"engine": model}
    else:
        raise ValueError("Invalid API")
    error_msg = None
    error_type = None
    for _ in range(max_retries):
        try:
            completion = openai.ChatCompletion.create(
                temperature=0.0, messages=messages, **model_dict
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
