import openai

from skllm.config import SKLLMConfig as _Config


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


def set_azure_credentials(key: str) -> None:
    """Sets OpenAI credentials for Azure.

    Parameters
    ----------
    key : str
        The OpenAI (Azure) key to use.
    """
    openai.api_type = "azure"
    openai.api_key = key
    openai.organization = None
    openai.api_base = _Config.get_azure_api_base()
    openai.api_version = _Config.get_azure_api_version()
