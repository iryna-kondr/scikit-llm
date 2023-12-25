from skllm.utils import retry
from vertexai.preview.language_models import ChatModel, TextGenerationModel


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
