import unittest
import json
from unittest.mock import patch, MagicMock
import numpy as np
from skllm.models.gpt_zero_shot_clf import ZeroShotGPTClassifier, MultiLabelZeroShotGPTClassifier

class TestZeroShotGPTClassifier(unittest.TestCase):

    @patch("skllm.models.gpt_zero_shot_clf.get_chat_completion", return_value=MagicMock())
    def test_fit_predict(self, mock_get_chat_completion):
        clf = ZeroShotGPTClassifier(openai_key="mock_key", openai_org="mock_org")  # Mock keys
        X = np.array(["text1", "text2", "text3"])
        y = np.array(["class1", "class2", "class1"])
        clf.fit(X, y)

        self.assertEqual(set(clf.classes_), set(["class1", "class2"]))
        self.assertEqual(clf.probabilities_, [2/3, 1/3])

        mock_get_chat_completion.return_value.choices[0].message = {"content": json.dumps({"label": "class1"})}
        predictions = clf.predict(X)

        self.assertEqual(predictions, ["class1", "class1", "class1"])

class TestMultiLabelZeroShotGPTClassifier(unittest.TestCase):

    @patch("skllm.models.gpt_zero_shot_clf.get_chat_completion", return_value=MagicMock())
    def test_fit_predict(self, mock_get_chat_completion):
        clf = MultiLabelZeroShotGPTClassifier(openai_key="mock_key", openai_org="mock_org")  # Mock keys
        X = np.array(["text1", "text2", "text3"])
        y = [["class1", "class2"], ["class1", "class2"], ["class1", "class2"]]  # Adjusted y to ensure [0.5, 0.5] probability
        clf.fit(X, y)

        self.assertEqual(set(clf.classes_), set(["class1", "class2"]))
        self.assertEqual(clf.probabilities_, [0.5, 0.5])

        mock_get_chat_completion.return_value.choices[0].message = {"content": json.dumps({"label": ["class1", "class2"]})}
        predictions = clf.predict(X)

        self.assertEqual(predictions, [["class1", "class2"], ["class1", "class2"], ["class1", "class2"]])

if __name__ == '__main__':
    unittest.main()
