from typing import Union, Optional

from skllm.prompts.templates import (
    FEW_SHOT_CLF_PROMPT_TEMPLATE,
    FEW_SHOT_MLCLF_PROMPT_TEMPLATE,
    FOCUSED_SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    TRANSLATION_PROMPT_TEMPLATE,
    ZERO_SHOT_CLF_PROMPT_TEMPLATE,
    ZERO_SHOT_MLCLF_PROMPT_TEMPLATE,
    EXPLAINABLE_NER_DENSE_PROMPT_TEMPLATE,
)

# TODO add validators


def build_zero_shot_prompt_slc(
    x: str, labels: str, template: str = ZERO_SHOT_CLF_PROMPT_TEMPLATE
) -> str:
    """Builds a prompt for zero-shot single-label classification.

    Parameters
    ----------
    x : str
        sample to classify
    labels : str
        candidate labels in a list-like representation
    template : str
        prompt template to use, must contain placeholders for all variables, by default ZERO_SHOT_CLF_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(x=x, labels=labels)


def build_few_shot_prompt_slc(
    x: str,
    labels: str,
    training_data: str,
    template: str = FEW_SHOT_CLF_PROMPT_TEMPLATE,
) -> str:
    """Builds a prompt for zero-shot single-label classification.

    Parameters
    ----------
    x : str
        sample to classify
    labels : str
        candidate labels in a list-like representation
    training_data : str
        training data to be used for few-shot learning
    template : str
        prompt template to use, must contain placeholders for all variables, by default ZERO_SHOT_CLF_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(x=x, labels=labels, training_data=training_data)


def build_few_shot_prompt_mlc(
    x: str,
    labels: str,
    training_data: str,
    max_cats: Union[int, str],
    template: str = FEW_SHOT_MLCLF_PROMPT_TEMPLATE,
) -> str:
    """Builds a prompt for few-shot single-label classification.

    Parameters
    ----------
    x : str
        sample to classify
    labels : str
        candidate labels in a list-like representation
    max_cats : Union[int,str]
        maximum number of categories to assign
    training_data : str
        training data to be used for few-shot learning
    template : str
        prompt template to use, must contain placeholders for all variables, by default ZERO_SHOT_CLF_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(
        x=x, labels=labels, training_data=training_data, max_cats=max_cats
    )


def build_zero_shot_prompt_mlc(
    x: str,
    labels: str,
    max_cats: Union[int, str],
    template: str = ZERO_SHOT_MLCLF_PROMPT_TEMPLATE,
) -> str:
    """Builds a prompt for zero-shot multi-label classification.

    Parameters
    ----------
    x : str
        sample to classify
    labels : str
        candidate labels in a list-like representation
    max_cats : Union[int,str]
        maximum number of categories to assign
    template : str
        prompt template to use, must contain placeholders for all variables, by default ZERO_SHOT_MLCLF_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(x=x, labels=labels, max_cats=max_cats)


def build_summary_prompt(
    x: str, max_words: Union[int, str], template: str = SUMMARY_PROMPT_TEMPLATE
) -> str:
    """Builds a prompt for text summarization.

    Parameters
    ----------
    x : str
        sample to summarize
    max_words : Union[int,str]
        maximum number of words to use in the summary
    template : str
        prompt template to use, must contain placeholders for all variables, by default SUMMARY_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(x=x, max_words=max_words)


def build_focused_summary_prompt(
    x: str,
    max_words: Union[int, str],
    focus: Union[int, str],
    template: str = FOCUSED_SUMMARY_PROMPT_TEMPLATE,
) -> str:
    """Builds a prompt for focused text summarization.

    Parameters
    ----------
    x : str
        sample to summarize
    max_words : Union[int,str]
        maximum number of words to use in the summary
    focus : Union[int,str]
        the topic(s) to focus on
    template : str
        prompt template to use, must contain placeholders for all variables, by default FOCUSED_SUMMARY_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(x=x, max_words=max_words, focus=focus)


def build_translation_prompt(
    x: str, output_language: str, template: str = TRANSLATION_PROMPT_TEMPLATE
) -> str:
    """Builds a prompt for text translation.

    Parameters
    ----------
    x : str
        sample to translate
    output_language : str
        language to translate to
    template : str
        prompt template to use, must contain placeholders for all variables, by default TRANSLATION_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(x=x, output_language=output_language)


def build_ner_prompt(
    entities: list,
    x: str,
    template: str = EXPLAINABLE_NER_DENSE_PROMPT_TEMPLATE,
) -> str:
    """Builds a prompt for named entity recognition.

    Parameters
    ----------
    entities : list
        list of entities to recognize
    x : str
        sample to recognize entities in
    template : str, optional
        prompt template to use, must contain placeholders for all variables, by default EXPLAINABLE_NER_DENSE_PROMPT_TEMPLATE

    Returns
    -------
    str
        prepared prompt
    """
    return template.format(entities=entities, x=x)
