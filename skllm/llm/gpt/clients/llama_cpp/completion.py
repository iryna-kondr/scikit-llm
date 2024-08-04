from skllm.llm.gpt.clients.llama_cpp.handler import ModelCache, LlamaHandler


def get_chat_completion(messages: dict, model: str, **kwargs):

    with ModelCache.lock:
        handler = ModelCache.get(model)
        if handler is None:
            handler = LlamaHandler(model)
            ModelCache.store(model, handler)

    return handler.get_chat_completion(messages, **kwargs)
