from pandas import DataFrame
from vertexai.language_models import TextGenerationModel


def tune(model: str, data: DataFrame, train_steps: int = 100):
    model = TextGenerationModel.from_pretrained(model)
    model.tune_model(
        training_data=data,
        train_steps=train_steps,
        tuning_job_location="europe-west4",  # the only supported training location atm
        tuned_model_location="us-central1",  # the only supported deployment location atm
    )
    return model
