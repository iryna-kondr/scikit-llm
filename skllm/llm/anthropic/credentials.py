from anthropic import Anthropic


def set_credentials(key: str) -> None:
    """Set the Anthropic key.

    Parameters
    ----------
    key : str
        The Anthropic key to use.
    """
    client = Anthropic(api_key=key)
    return client
