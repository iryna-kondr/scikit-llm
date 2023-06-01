import openai

def set_credentials(key: str, org: str) -> None:
    """
    Set the OpenAI key and organization.

    Parameters
    ----------
    key : str
        The OpenAI key to use.
    org : str
        The OpenAI organization to use.
    """
    openai.api_key = key
    openai.organization = org
