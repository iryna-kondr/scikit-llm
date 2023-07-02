from time import sleep

from vertexai.preview.language_models import ChatModel, TextGenerationModel

# TODO reduce code duplication for retrying logic


def get_completion(model: str, text: str, max_retries: int = 3):
    for _ in range(max_retries):
        try:
            if model.startswith("text-"):
                model = TextGenerationModel.from_pretrained(model)
            else:
                model = TextGenerationModel.get_tuned_model(model)
            response = model.predict(text, temperature=0.0)
            return response.text
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            sleep(3)
    print(
        f"Could not obtain the completion after {max_retries} retries: `{error_type} ::"
        f" {error_msg}`"
    )


def get_completion_chat_mode(model: str, context: str, text: str, max_retries: int = 3):
    for _ in range(max_retries):
        try:
            model = ChatModel.from_pretrained(model)
            chat = model.start_chat(context=context)
            response = chat.send_message(text, temperature=0.0)
            return response.text
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            sleep(3)
    print(
        f"Could not obtain the completion after {max_retries} retries: `{error_type} ::"
        f" {error_msg}`"
    )
