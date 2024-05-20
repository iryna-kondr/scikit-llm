from skllm.utils import retry
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.generative_models import GenerativeModel, GenerationConfig


@retry(max_retries=3)
def get_completion(model: str, text: str):
    if model.startswith("text-"):
        model_instance = TextGenerationModel.from_pretrained(model)
    else:
        model_instance = TextGenerationModel.get_tuned_model(model)
    response = model_instance.predict(text, temperature=0.0)
    return response.text


@retry(max_retries=3)
def get_completion_chat_mode(model: str, context: str, text: str):
    model_instance = ChatModel.from_pretrained(model)
    chat = model_instance.start_chat(context=context)
    response = chat.send_message(text, temperature=0.0)
    return response.text


@retry(max_retries=3)
def get_completion_chat_gemini(model: str, context: str, text: str):
    model_instance = GenerativeModel(model, system_instruction=context)
    response = model_instance.generate_content(
        text, generation_config=GenerationConfig(temperature=0.0)
    )
    return response.text
