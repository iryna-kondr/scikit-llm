def get_zero_shot_prompt_slc(x, labels):
    lines = [
        "You will be provided with the following information:",
        "1. An arbitrary text sample. The sample is delimited with triple backticks.",
        "2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.",
        "",
        "Perform the following tasks:",
        "1. Identify to which category the provided text belongs to with the highest probability.",
        "2. Assign the provided text to that category.",
        "3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.",
        "\n", 
        f"List of categories: {repr(labels)}"
        "\n",
        f"Text sample: ```{x}```",
        "\n",
        "Your JSON response: "
    ]
    prompt = "\n".join(lines)
    return prompt

def get_zero_shot_prompt_mlc(x, labels, max_cats):
    lines = [
        "You will be provided with the following information:",
        "1. An arbitrary text sample. The sample is delimited with triple backticks.",
        f"2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated. The text sample belongs to at least one category but cannot exceed {max_cats}.",
        "",
        "Perform the following tasks:",
        "1. Identify to which categories the provided text belongs to with the highest probability.",
        f"2. Assign the text sample to at least 1 but up to {max_cats} categories based on the probabilities.",
        "3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the list of assigned categories. Do not provide any additional information except the JSON."
        "\n", 
        f"List of categories: {repr(labels)}"
        "\n",
        f"Text sample: ```{x}```",
        "\n",
        "Your JSON response: "
    ]
    prompt = "\n".join(lines)
    return prompt