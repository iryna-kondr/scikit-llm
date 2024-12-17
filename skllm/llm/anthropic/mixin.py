from typing import Optional, Union, Any, List, Dict, Mapping
from skllm.config import SKLLMConfig as _Config
from skllm.llm.anthropic.completion import get_chat_completion
from skllm.utils import extract_json_key
from skllm.llm.base import BaseTextCompletionMixin, BaseClassifierMixin
import json


class ClaudeMixin:
    """A mixin class that provides Claude API key to other classes."""

    _prefer_json_output = False

    def _set_keys(self, key: Optional[str] = None) -> None:
        """Set the Claude API key."""
        self.key = key

    def _get_claude_key(self) -> str:
        """Get the Claude key from the class or config file."""
        key = self.key
        if key is None:
            key = _Config.get_anthropic_key()
        if key is None:
            raise RuntimeError("Claude API key was not found")
        return key

class ClaudeTextCompletionMixin(ClaudeMixin, BaseTextCompletionMixin):
    """A mixin class that provides text completion capabilities using the Claude API."""

    def _get_chat_completion(
        self,
        model: str,
        messages: Union[str, List[Dict[str, str]]],
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """Gets a chat completion from the Anthropic API.

        Parameters
        ----------
        model : str
            The model to use.
        messages : Union[str, List[Dict[str, str]]]
            input messages to use.
        system_message : Optional[str]
            A system message to use.
        **kwargs : Any
            placeholder.

        Returns
        -------
        completion : dict
        """
        if isinstance(messages, str):
            messages = [{"content": messages}]
        elif isinstance(messages, list):
            messages = [{"content": msg["content"]} for msg in messages]
        
        completion = get_chat_completion(
            messages=messages,
            key=self._get_claude_key(),
            model=model,
            system=system_message,
            json_response=self._prefer_json_output,
            **kwargs,
        )
        return completion

    def _convert_completion_to_str(self, completion: Mapping[str, Any]):
        """Converts Claude API completion to string."""
        try:
            if hasattr(completion, 'content'):
                return completion.content[0].text
            return completion.get('content', [{}])[0].get('text', '')
        except Exception as e:
            print(f"Error converting completion to string: {str(e)}")
            return ""

class ClaudeClassifierMixin(ClaudeTextCompletionMixin, BaseClassifierMixin):
    """A mixin class that provides classification capabilities using Claude API."""
    
    _prefer_json_output = True
    
    def _extract_out_label(self, completion: Mapping[str, Any], **kwargs) -> str:
        """Extracts the label from a Claude API completion."""
        try:
            content = self._convert_completion_to_str(completion)           
            if not self._prefer_json_output:
                return content.strip()
                
            # Attempt to parse content as JSON and extract label  
            try:
                data = json.loads(content)
                if "label" in data:
                    return data["label"]
            except json.JSONDecodeError:
                pass
            return ""      
        
        except Exception as e:
            print(f"Error extracting label: {str(e)}")
            return ""
        