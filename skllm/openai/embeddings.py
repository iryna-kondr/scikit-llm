from time import sleep

import openai

from skllm.openai.credentials import set_credentials


def get_embedding(
    text: str, key: str, org: str, model: str="text-embedding-ada-002", max_retries: int=3
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
    
    Returns
    -------
    emb : list
        The GPT embedding for the string.
    """
    set_credentials(key, org)
    text = text.replace("\n", " ")
    error_msg = None
    error_type = None
    for _ in range(max_retries):
        try:
            emb = openai.Embedding.create(input=[text], model=model)["data"][0][
                "embedding"
            ]
            if not isinstance(emb, list):
                raise ValueError(f"Encountered unknown embedding format. Expected list, got {type(emb)}")
            return emb
        except Exception as e:
            error_msg = str(e)
            error_type =  type(e).__name__
            sleep(3)
    raise RuntimeError(
        f"Could not obtain the embedding after {max_retries} retries: `{error_type} :: {error_msg}`"
    )
