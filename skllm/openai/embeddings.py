import openai
from time import sleep
from skllm.openai.credentials import set_credentials

def get_embedding(
    text, key: str, org: str, model="text-embedding-ada-002", max_retries=3
):
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
