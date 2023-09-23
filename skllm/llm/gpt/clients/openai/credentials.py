import openai
from skllm.config import SKLLMConfig as _Config
from time import sleep


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
        set_azure_credentials(key, org)
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


def set_credentials(key: str, org: str) -> None:
    """Set the OpenAI key and organization.

    Parameters
    ----------
    key : str
        The OpenAI key to use.
    org : str
        The OPEN AI organization ID to use.
    """
    openai.api_key = key
    openai.organization = org
    openai.api_type = "open_ai"
    openai.api_version = None
    openai.api_base = "https://api.openai.com/v1"


def set_azure_credentials(key: str, org: str) -> None:
    """Sets OpenAI credentials for Azure.

    Parameters
    ----------
    key : str
        The OpenAI (Azure) key to use.
    org : str
        The OpenAI (Azure) organization ID to use.
    """
    if not openai.api_type or not openai.api_type.startswith("azure"):
        openai.api_type = "azure"
    openai.api_key = key
    openai.organization = org
    openai.api_base = _Config.get_azure_api_base()
    openai.api_version = _Config.get_azure_api_version()
