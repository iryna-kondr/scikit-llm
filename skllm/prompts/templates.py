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
