import openai
from skllm.config import SKLLMConfig as _Config
from time import sleep
from openai import OpenAI, AzureOpenAI
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
    url = _Config.get_gpt_url()
    client = OpenAI(api_key=key, organization=org, base_url=url)
    return client


def set_azure_credentials(key: str, org: str) -> None:
    """Sets OpenAI credentials for Azure.

    Parameters
    ----------
    key : str
        The OpenAI (Azure) key to use.
    org : str
        The OpenAI (Azure) organization ID to use.
    """
    client = AzureOpenAI(
        api_key=key,
        organization=org,
        api_version=_Config.get_azure_api_version(),
        azure_endpoint=_Config.get_azure_api_base(),
    )
    return client
