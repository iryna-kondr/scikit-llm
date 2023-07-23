<p align="center">
  <img src="https://github.com/iryna-kondr/scikit-llm/blob/main/logo.png?raw=true" max-height="200"/>
</p>

# Scikit-LLM: Sklearn Meets Large Language Models

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

## Our Related Projects 🔗

<a href="https://github.com/OKUA1/agent_dingo"><img src="https://gist.githubusercontent.com/OKUA1/6264a95a8abd225c74411a2b707b0242/raw/1b231aab718fcab624faa33d9c10d0eee17ca160/dingo_light.svg"/></a> <br>
<a href="https://github.com/OKUA1/falcon"><img src="https://raw.githubusercontent.com/gist/OKUA1/6264a95a8abd225c74411a2b707b0242/raw/3cedb53538cb04656cd9d7d07e697e726896ce9f/falcon_light.svg"/></a>

## Documentation 📚

### Configuring OpenAI API Key

At the moment the majority of the Scikit-LLM estimators are only compatible with some of the OpenAI models. Hence, a user-provided OpenAI API key is required.

```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANISATION>")
```

**Important notice:**

- If you have a free trial OpenAI account, the [rate limits](https://platform.openai.com/docs/guides/rate-limits/overview) are not sufficient (specifically 3 requests per minute). Please switch to the "pay as you go" plan first.
- When calling `SKLLMConfig.set_openai_org`, you have to provide your organization ID and **NOT** the name. You can find your ID [here](https://platform.openai.com/account/org-settings).

### Using Azure OpenAI

```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("<YOUR_KEY>")  # use azure key instead
SKLLMConfig.set_azure_api_base("<API_BASE>")

# start with "azure::" prefix when setting the model name
model_name = "azure::<model_name>"
# e.g. ZeroShotGPTClassifier(openai_model="azure::gpt-3.5-turbo")
```

Note: Azure OpenAI is not supported by the preprocessors at the moment.

### Using GPT4ALL

In addition to OpenAI, some of the models can use [gpt4all](https://gpt4all.io/index.html) as a backend.

**This feature is considered higly experimental!**

In order to use gpt4all, you need to install the corresponding submodule:

```bash
pip install "scikit-llm[gpt4all]"
```

In order to switch from OpenAI to GPT4ALL model, simply provide a string of the format `gpt4all::<model_name>` as an argument. While the model runs completely locally, the estimator still treats it as an OpenAI endpoint and will try to check that the API key is present. You can provide any string as a key.

```python
SKLLMConfig.set_openai_key("any string")
SKLLMConfig.set_openai_org("any string")

ZeroShotGPTClassifier(openai_model="gpt4all::ggml-model-gpt4all-falcon-q4_0.bin")
```

When running for the first time, the model file will be downloaded automatially.

When using gpt4all please keep the following in mind:

1. Not all gpt4all models are commercially licensable, please consult gpt4all website for more details.
2. The accuracy of the models may be much lower compared to ones provided by OpenAI (especially gpt-4).
3. Not all of the available models were tested, some may not work with scikit-llm at all.

### Supported models by a non-standard backend

At the moment only the following estimators support non-standard backends (gpt4all, azure):

- `ZeroShotGPTClassifier`
- `MultiLabelZeroShotGPTClassifier`
- `FewShotGPTClassifier`

### Zero-Shot Text Classification

One of the powerful ChatGPT features is the ability to perform text classification without being re-trained. For that, the only requirement is that the labels must be descriptive.

We provide a class `ZeroShotGPTClassifier` that allows to create such a model as a regular scikit-learn classifier.

Example 1: Training as a regular classifier

```python
from skllm import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()

clf = ZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf.fit(X, y)
labels = clf.predict(X)
```

Scikit-LLM will automatically query the OpenAI API and transform the response into a regular list of labels.

Additionally, Scikit-LLM will ensure that the obtained response contains a valid label. If this is not the case, a label will be selected randomly (label probabilities are proportional to label occurrences in the training set).

Example 2: Training without labeled data

Since the training data is not strictly required, it can be fully ommited. The only thing that has to be provided is the list of candidate labels.

```python
from skllm import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset

X, _ = get_classification_dataset()

clf = ZeroShotGPTClassifier()
clf.fit(None, ["positive", "negative", "neutral"])
labels = clf.predict(X)
```

**Note:** unlike in a typical supervised setting, the performance of a zero-shot classifier greatly depends on how the label itself is structured. It has to be expressed in natural language, be descriptive and self-explanatory. For example, in the previous semantic classification task, it could be beneficial to transform a label from `"<semantics>"` to `"the semantics of the provided text is <semantics>"`.

### Multi-Label Zero-Shot Text Classification

With a class `MultiLabelZeroShotGPTClassifier` it is possible to perform the classification in multi-label setting, which means that each sample might be assigned to one or several distinct classes.

Example:

```python
from skllm import MultiLabelZeroShotGPTClassifier
from skllm.datasets import get_multilabel_classification_dataset

X, y = get_multilabel_classification_dataset()

clf = MultiLabelZeroShotGPTClassifier(max_labels=3)
clf.fit(X, y)
labels = clf.predict(X)
```

Similarly to the `ZeroShotGPTClassifier` it is sufficient if only candidate labels are provided. However, this time the classifier expects `y` of a type `List[List[str]]`.

```python
from skllm import MultiLabelZeroShotGPTClassifier
from skllm.datasets import get_multilabel_classification_dataset

X, _ = get_multilabel_classification_dataset()
candidate_labels = [
    "Quality",
    "Price",
    "Delivery",
    "Service",
    "Product Variety",
    "Customer Support",
    "Packaging",
    "User Experience",
    "Return Policy",
    "Product Information",
]
clf = MultiLabelZeroShotGPTClassifier(max_labels=3)
clf.fit(None, [candidate_labels])
labels = clf.predict(X)
```

### Few-Shot Text Classification

With `FewShotGPTClassifier` it is possible to perform a few-shot classification, which means that the training samples will be added to prompt and passed to the model.

```python
from skllm import FewShotGPTClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = FewShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf.fit(X, y)
labels = clf.predict(X)
```

While the api remains the same as for the zero shot classifier, there are a few things to take into account:

- the "training" requires some labelled training data;
- the training set should be small enough to fit into a single prompt (we recommend up to 10 samples per label);
- because of the significantly larger prompt, the inference takes longer and consumes higher amount of tokens.

Note: as the model is not being re-trained, but uses the training data during inference, one could say that this is still a (different) zero-shot approach.

### Dynamic Few-Shot Text Classification

`DynamicFewShotGPTClassifier` dynamically selects N samples per class to include in the prompt. This allows the few-shot classifier to scale to datasets that are too large for the standard context window of LLMs.

*How does it work?*

During fitting, the whole dataset is partitioned by class, vectorized, and stored.

During inference, the [annoy](https://github.com/spotify/annoy) library is used for fast neighbor lookup, which allows including only the most similar examples in the prompt.

```python
from skllm import DynamicFewShotGPTClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = DynamicFewShotGPTClassifier(n_examples=3)
clf.fit(X, y)
labels = clf.predict(X)
```

### Text Classification with Google PaLM 2

At the moment 3 PaLM based models are available in test mode:

- `ZeroShotPaLMClassifier` - zero-shot text classification with PaLM 2;
- `PaLMClassifier` - fine-tunable text classifier with PaLM 2;
- `PaLM` - fine-tunable estimator that can be trained on arbitrary text input-output pairs.

Example:

```python
from skllm.models.palm import PaLMClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = PaLMClassifier(n_update_steps=100)
clf.fit(X, y)
labels = clf.predict(X)
```

A more detailed documentation will follow soon. For now, please refer to our [official guide on Medium](https://medium.com/@iryna230520/fine-tune-google-palm-2-with-scikit-llm-d41b0aa673a5).

### Text Vectorization

As an alternative to using GPT as a classifier, it can be used solely for data preprocessing. `GPTVectorizer` allows to embed a chunk of text of arbitrary length to a fixed-dimensional vector, that can be used with virtually any classification or regression model.

Example 1: Embedding the text

```python
from skllm.preprocessing import GPTVectorizer

model = GPTVectorizer()
vectors = model.fit_transform(X)
```

Example 2: Combining the Vectorizer with the XGBoost Classifier in a Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

steps = [("GPT", GPTVectorizer()), ("Clf", XGBClassifier())]
clf = Pipeline(steps)
clf.fit(X_train, y_train_encoded)
yh = clf.predict(X_test)
```

### Text Summarization

GPT excels at performing summarization tasks. Therefore, we provide `GPTSummarizer` that can be used both as stand-alone estimator, or as a preprocessor (in this case we can make an analogy with a dimensionality reduction preprocessor).

Example:

```python
from skllm.preprocessing import GPTSummarizer
from skllm.datasets import get_summarization_dataset

X = get_summarization_dataset()
s = GPTSummarizer(openai_model="gpt-3.5-turbo", max_words=15)
summaries = s.fit_transform(X)
```

Please be aware that the `max_words` hyperparameter sets a soft limit, which is not strictly enforced outside of the prompt. Therefore, in some cases, the actual number of words might be slightly higher.

It is possible to generate a summary, emphasizing a specific concept, by providing an optional parameter `focus`:

```python
s = GPTSummarizer(openai_model="gpt-3.5-turbo", max_words=15, focus="apples")
```

### Text Translation

GPT models have demonstrated their effectiveness in translation tasks by generating accurate translations across various languages. Thus, we added `GPTTranslator` that allows translating an arbitraty text into a language of interest.

Example:

```python
from skllm.preprocessing import GPTTranslator
from skllm.datasets import get_translation_dataset

X = get_translation_dataset()
t = GPTTranslator(openai_model="gpt-3.5-turbo", output_language="English")
translated_text = t.fit_transform(X)
```
