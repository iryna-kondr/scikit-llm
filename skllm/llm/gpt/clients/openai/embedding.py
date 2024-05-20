from skllm.llm.gpt.clients.openai.credentials import set_credentials, set_azure_credentials
from skllm.utils import retry
import openai
from openai import OpenAI


@retry(max_retries=3)
def get_embedding(
    text: str,
    key: str,
    org: str,
    model: str = "text-embedding-ada-002",
    api: str = "openai"
):
    """
    Encodes a string and return the embedding for a string.

    Parameters
    ----------
    text : str
        The string to encode.
    key : str
        The OPEN AI key to use.
    org : str
        The OPEN AI organization ID to use.
    model : str, optional
        The model to use. Defaults to "text-embedding-ada-002".
    max_retries : int, optional
        The maximum number of retries to use. Defaults to 3.
    api: str, optional
        The API to use. Must be one of "openai" or "azure". Defaults to "openai".

    Returns
    -------
    emb : list
        The GPT embedding for the string.
    """
    if api in ("openai", "custom_url"):
        client = set_credentials(key, org)
    elif api == "azure":
        client = set_azure_credentials(key, org)
    text = [str(t).replace("\n", " ") for t in text]
    embeddings = []
    emb = client.embeddings.create(input=text, model=model)
    for i in range(len(emb.data)):
        e = emb.data[i].embedding
        if not isinstance(e, list):
            raise ValueError(
                f"Encountered unknown embedding format. Expected list, got {type(emb)}"
            )
        embeddings.append(e)
    return embeddings
