<div align="center">
  <a href="https://github.com/iryna-kondr/scikit-llm">
    <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/43d26803ba9cccd1ea478bb491fd9b83/raw/e3a5e7759f508a145fa35b204ed363164adabeca/skllm_icon_color.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/43d26803ba9cccd1ea478bb491fd9b83/raw/e3a5e7759f508a145fa35b204ed363164adabeca/skllm_icon_white.svg">
  <img alt="Hashnode logo" src="https://gist.githubusercontent.com/OKUA1/43d26803ba9cccd1ea478bb491fd9b83/raw/e3a5e7759f508a145fa35b204ed363164adabeca/skllm_icon_color.svg" height = "220">
</picture>
</a>
</div>

# Scikit-LLM: Scikit-Learn Meets Large Language Models

Seamlessly integrate powerful language models like ChatGPT into scikit-learn for enhanced text analysis tasks.

## Installation 💾

```bash
pip install scikit-llm
```

## Support us 🤝

You can support the project in the following ways:

- ⭐ Star Scikit-LLM on GitHub (click the star button in the top right corner)
- 💡 Provide your feedback or propose ideas in the [issues](https://github.com/iryna-kondr/scikit-llm/issues) section or [Discord](https://discord.gg/YDAbwuWK7V)
- 📰 Post about Scikit-LLM on LinkedIn or other platforms
- 🔗 Check out our other projects (cards below are clickable):

<a href="https://github.com/OKUA1/agent_dingo"><img src="https://gist.githubusercontent.com/OKUA1/6264a95a8abd225c74411a2b707b0242/raw/1b231aab718fcab624faa33d9c10d0eee17ca160/dingo_light.svg"/></a> <br>
<a href="https://github.com/OKUA1/falcon"><img src="https://raw.githubusercontent.com/gist/OKUA1/6264a95a8abd225c74411a2b707b0242/raw/3cedb53538cb04656cd9d7d07e697e726896ce9f/falcon_light.svg"/></a>

## Quick Start & Documentation 📚

Quick start example of zero-shot text classification using GPT:

```python
# Import the necessary modules
from skllm.datasets import get_classification_dataset
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

# Configure the credentials
SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")

# Load a demo dataset
X, y = get_classification_dataset() # labels: positive, negative, neutral

# Initialize the model and make the predictions
clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(X,y)
clf.predict(X)
```

For more information please refer to the [documentation](https://beastbyteai.github.io/scikit-llm-docs/).

## Citation

You can cite Scikit-LLM using the following BibTeX:

```
@software{ScikitLLM,
  author = {Iryna Kondrashchenko and Oleh Kostromin},
  year = {2023},
  publisher = {beastbyte.ai},
  address = {Linz, Austria},
  title = {Scikit-LLM: Scikit-Learn Meets Large Language Models},
  url = {https://github.com/iryna-kondr/scikit-llm }
}
```
