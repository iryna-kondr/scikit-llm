import unittest
from unittest.mock import patch
import json
from skllm.llm.anthropic.mixin import (
    ClaudeMixin,
    ClaudeTextCompletionMixin,
    ClaudeClassifierMixin,
)


class TestClaudeMixin(unittest.TestCase):
    def test_ClaudeMixin(self):
        mixin = ClaudeMixin()
        mixin._set_keys("test_key")
        self.assertEqual(mixin._get_claude_key(), "test_key")


class TestClaudeTextCompletionMixin(unittest.TestCase):
    @patch("skllm.llm.anthropic.mixin.get_chat_completion")
    def test_chat_completion_with_valid_params(self, mock_get_chat_completion):
        mixin = ClaudeTextCompletionMixin()
        mixin._set_keys("test_key")
        
        mock_get_chat_completion.return_value = {
            "content": [
                {"type": "text", "text": "test response"}
            ]
        }

        completion = mixin._get_chat_completion(
            model="claude-3-haiku-20240307",
            messages="Hello",
            system_message="Test system"
        )

        self.assertEqual(
            mixin._convert_completion_to_str(completion),
            "test response"
        )
        mock_get_chat_completion.assert_called_once()


class TestClaudeClassifierMixin(unittest.TestCase):
    @patch("skllm.llm.anthropic.mixin.get_chat_completion")
    def test_extract_out_label_with_valid_completion(self, mock_get_chat_completion):
        mixin = ClaudeClassifierMixin()
        mixin._set_keys("test_key")
        
        mock_get_chat_completion.return_value = {
            "content": [
                {"type": "text", "text": '{"label":"hello world"}'}
            ]
        }

        completion = mixin._get_chat_completion(
            model="claude-3-haiku-20240307",
            messages="Hello",
            system_message="World"
        )
        self.assertEqual(mixin._extract_out_label(completion), "hello world")
        mock_get_chat_completion.assert_called_once()