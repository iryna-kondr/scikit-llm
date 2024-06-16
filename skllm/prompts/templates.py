ZERO_SHOT_CLF_PROMPT_TEMPLATE = """
You will be provided with the following information:
1. An arbitrary text sample. The sample is delimited with triple backticks.
2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.

Perform the following tasks:
1. Identify to which category the provided text belongs to with the highest probability.
2. Assign the provided text to that category.
3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

List of categories: {labels}

Text sample: ```{x}```

Your JSON response:
"""

ZERO_SHOT_CLF_SHORT_PROMPT_TEMPLATE = """
Classify the following text into one of the following classes: {labels}. Provide your response in a JSON format containing a single key `label`.
Text: ```{x}```
"""

ZERO_SHOT_MLCLF_SHORT_PROMPT_TEMPLATE = """
Classify the following text into at least 1 but up to {max_cats} of the following classes: {labels}. Provide your response in a JSON format containing a single key `label`.
Text: ```{x}```
"""

FEW_SHOT_CLF_PROMPT_TEMPLATE = """
You will be provided with the following information:
1. An arbitrary text sample. The sample is delimited with triple backticks.
2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
3. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data.

Perform the following tasks:
1. Identify to which category the provided text belongs to with the highest probability.
2. Assign the provided text to that category.
3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

List of categories: {labels}

Training data:
{training_data}

Text sample: ```{x}```

Your JSON response:
"""

FEW_SHOT_MLCLF_PROMPT_TEMPLATE = """
You will be provided with the following information:
1. An arbitrary text sample. The sample is delimited with triple backticks.
2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
3. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data.

Perform the following tasks:
1. Identify to which category the provided text belongs to with the highest probability.
2. Assign the text sample to at least 1 but up to {max_cats} categories based on the probabilities.
3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the array of assigned categories. Do not provide any additional information except the JSON.

List of categories: {labels}

Training data:
{training_data}

Text sample: ```{x}```

Your JSON response:
"""

ZERO_SHOT_MLCLF_PROMPT_TEMPLATE = """
You will be provided with the following information:
1. An arbitrary text sample. The sample is delimited with triple backticks.
2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated. The text sample belongs to at least one category but cannot exceed {max_cats}.

Perform the following tasks:
1. Identify to which categories the provided text belongs to with the highest probability.
2. Assign the text sample to at least 1 but up to {max_cats} categories based on the probabilities.
3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the array of assigned categories. Do not provide any additional information except the JSON.

List of categories: {labels}

Text sample: ```{x}```

Your JSON response:
"""

SUMMARY_PROMPT_TEMPLATE = """
Your task is to generate a summary of the text sample.
Summarize the text sample provided below, delimited by triple backticks, in at most {max_words} words.

Text sample: ```{x}```
Summarized text:
"""

FOCUSED_SUMMARY_PROMPT_TEMPLATE = """
As an input you will receive:
1. A focus parameter delimited with square brackets.
2. A single text sample delimited with triple backticks.

Perform the following actions:
1. Determine whether there is something in the text that matches focus. Do not output anything.
2. Summarise the text in at most {max_words} words.
3. If possible, make the summarisation focused on the concept provided in the focus parameter. Otherwise, provide a general summarisation. Do not state that general summary is provided.
4. Do not output anything except of the summary. Do not output any text that was not present in the original text.
5. If no focused summary possible, or the mentioned concept is not present in the text, output "Mentioned concept is not present in the text." and the general summary. Do not state that general summary is provided.

Focus: [{focus}]

Text sample: ```{x}```

Summarized text:
"""

TRANSLATION_PROMPT_TEMPLATE = """
If the original text, delimited by triple backticks, is already in {output_language} language, output the original text.
Otherwise, translate the original text, delimited by triple backticks, to {output_language} language, and output the translated text only. Do not output any additional information except the translated text.

Original text: ```{x}```
Output:
"""

NER_SYSTEM_MESSAGE_TEMPLATE = """You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a text provided by the user. 
Mark the entities with tags according to the following guidelines:
    - Use XML format to tag entities; 
    - All entities must be enclosed in <entity>...</entity> tags; All other text must be enclosed in <not_entity>...</not_entity> tags; No content should be outside of these tags;
    - The tagging operation must be invertible, i.e. the original text must be recoverable from the tagged textl; This is crucial and easy to overlook, double-check this requirement;
    - Adjacent entities should be separated into different tags;
    - The list of entities is strictly restricted to the following: {entities}.
"""

NER_SYSTEM_MESSAGE_SPARSE = """You are an expert in Natural Language Processing."""

EXPLAINABLE_NER_DENSE_PROMPT_TEMPLATE = """You are provided with a text. Your task is to identify and tag all named entities within the text using the following entity types only:
{entities}

For each entity, provide a brief explanation for your choice within an XML comment. Use the following XML tag format for each entity:

<entity><reasoning>Your reasoning here</reasoning><tag>ENTITY_NAME_UPPERCASE</tag><value>Entity text</value></entity>

The remaining text must be enclosed in a <not_entity>TEXT</not_entity> tag.

Focus on the context and meaning of each entity rather than just the exact words. The tags should encompass the entire entity based on its definition and usage in the sentence. It is crucial to base your decision on the description of the entity, not just its name.

Format example:

Input:
```This text contains some entity and another entity.```

Output:
```xml
<not_entity>This text contains </not_entity><entity><reasoning>some justification</reasoning><tag>ENTITY1</tag><value>some entity</value></entity><not_entity> and another </not_entity><entity><reasoning>another justification</reasoning><tag>ENTITY2</tag><value>entity</value></entity><not_entity>.</not_entity>
```

Input:
```
{x}
```

Output (origina text with tags):
"""


EXPLAINABLE_NER_SPARSE_PROMPT_TEMPLATE = """You are provided with a text. Your task is to identify and tag all named entities within the text using the following entity types only:
{entities}

You must provide the following information for each entity:
- The reasoning of why you tagged the entity as such; Based on the reasoning, a non-expert should be able to evaluate your decision;
- The tag of the entity (uppercase);
- The value of the entity (as it appears in the text).

Your response should be json formatted using the following schema:

{{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "array",
  "items": [
    {{
      "type": "object",
      "properties": {{
        "reasoning": {{
          "type": "string"
        }},
        "tag": {{
          "type": "string"
        }},
        "value": {{
          "type": "string"
        }}
      }},
      "required": [
        "reasoning",
        "tag",
        "value"
      ]
    }}
  ]
}}


Input:
```
{x}
```

Output json:
"""

