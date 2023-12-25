import unittest
from unittest.mock import patch
import json
from skllm.llm.gpt.mixin import (
    construct_message,
    _build_clf_example,
    GPTMixin,
    GPTTextCompletionMixin,
    GPTClassifierMixin,
)


class TestGPTMixin(unittest.TestCase):
    def test_construct_message(self):
        self.assertEqual(
            construct_message("user", "Hello"), {"role": "user", "content": "Hello"}
        )
        with self.assertRaises(ValueError):
            construct_message("invalid_role", "Hello")

    def test_build_clf_example(self):
        x = "Hello"
        y = "Hi"
        system_msg = "You are a text classification model."
        expected_output = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": x},
                    {"role": "assistant", "content": y},
                ]
            }
        )
        self.assertEqual(_build_clf_example(x, y, system_msg), expected_output)

    def test_GPTMixin(self):
        mixin = GPTMixin()
        mixin._set_keys("test_key", "test_org")
        self.assertEqual(mixin._get_openai_key(), "test_key")
        self.assertEqual(mixin._get_openai_org(), "test_org")


class TestGPTTextCompletionMixin(unittest.TestCase):
    @patch("skllm.llm.gpt.mixin.get_chat_completion")
    def test_chat_completion_with_valid_params(self, mock_get_chat_completion):
        # Setup
        mixin = GPTTextCompletionMixin()
        mixin._set_keys("test_key", "test_org")
        mock_get_chat_completion.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }

        _ = mixin._get_chat_completion("test-model", "Hello", "World")

        mock_get_chat_completion.assert_called_once()


class TestGPTClassifierMixin(unittest.TestCase):
    @patch("skllm.llm.gpt.mixin.get_chat_completion")
    def test_extract_out_label_with_valid_completion(self, mock_get_chat_completion):
        mixin = GPTClassifierMixin()
        mixin._set_keys("test_key", "test_org")
        mock_get_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"label":"hello world"}'}}]
        }
        res = mixin._get_chat_completion("test-model", "Hello", "World")
        self.assertEqual(mixin._extract_out_label(res), "hello world")
        mock_get_chat_completion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
