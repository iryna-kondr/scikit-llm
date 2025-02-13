from typing import Dict, List, Optional
from skllm.llm.anthropic.credentials import set_credentials
from skllm.utils import retry
from model_constants import ANTHROPIC_CLAUDE_MODEL

@retry(max_retries=3)
def get_chat_completion(
    messages: List[Dict],
    key: str,
    model: str = ANTHROPIC_CLAUDE_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    system: Optional[str] = None,
    json_response: bool = False,
) -> dict:
    """
    Gets a chat completion from the Anthropic Claude API using the Messages API.

    Parameters
    ----------
    messages : dict
        Input messages to use.
    key : str
        The Anthropic API key to use.
    model : str, optional
        The Claude model to use.
    max_tokens : int, optional
        Maximum tokens to generate.
    temperature : float, optional
        Sampling temperature.
    system : str, optional
        System message to set the assistant's behavior.
    json_response : bool, optional
        Whether to request a JSON-formatted response. Defaults to False.

    Returns
    -------
    response : dict
        The completion response from the API.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    if not isinstance(messages, list):
        raise TypeError("Messages must be a list")
    
    client = set_credentials(key)
    
    if json_response and system:
        system = f"{system.rstrip('.')}. Respond in JSON format."
    elif json_response:
        system = "Respond in JSON format."

    formatted_messages = [
        {
            "role": "user", # Explicitly set role to "user"
            "content": [
                {
                    "type": "text",
                    "text": message.get("content", "")
                }
            ]
        }
        for message in messages
    ]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=formatted_messages,
    )
    return response