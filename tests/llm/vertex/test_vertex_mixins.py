import unittest
from unittest.mock import patch
import json
from skllm.llm.vertex.mixin import *


class TestVertexTextCompletionMixin(unittest.TestCase):
    @patch("skllm.llm.vertex.mixin.get_completion")
    def test_chat_completion_with_valid_params(self, mock_get_chat_completion):
        # Setup
        mixin = VertexTextCompletionMixin()
        mock_get_chat_completion.return_value = "res"
        _ = mixin._get_chat_completion("test-model", "Hello", "World")
        mock_get_chat_completion.assert_called_once()


class TestVertexClassifierMixin(unittest.TestCase):
    @patch("skllm.llm.vertex.mixin.get_completion")
    def test_extract_out_label_with_valid_completion(self, mock_get_chat_completion):
        mixin = VertexClassifierMixin()
        mock_get_chat_completion.return_value = '{"label":"hello world"}'
        res = mixin._get_chat_completion("test-model", "Hello", "World")
        self.assertEqual(mixin._extract_out_label(res), "hello world")
        mock_get_chat_completion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
