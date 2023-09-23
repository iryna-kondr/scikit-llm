from typing import Optional
import openai
from time import sleep
from datetime import datetime
import os


def create_tuning_job(
    model: str,
    training_file: str,
    n_epochs: Optional[str] = None,
    suffix: Optional[str] = None,
):
    out = openai.File.create(file=open(training_file, "rb"), purpose="fine-tune")
    print(f"Created new file. FILE_ID = {out['id']}")
    print(f"Waiting for file to be processed ...")
    while not wait_file_ready(out["id"]):
        sleep(5)
    # delete the training_file after it is uploaded
    os.remove(training_file)
    params = {
        "model": model,
        "training_file": out["id"],
    }
    if n_epochs is not None:
        params["hyperparameters"] = {"n_epochs": n_epochs}
    if suffix is not None:
        params["suffix"] = suffix
    return openai.FineTuningJob.create(**params)


def await_results(job_id: str, check_interval: int = 120):
    while True:
        job = openai.FineTuningJob.retrieve(job_id)
        status = job["status"]
        if status == "succeeded":
            return job
        elif status == "failed" or status == "cancelled":
            print(job)
            raise RuntimeError(f"Tuning job failed with status {status}")
        else:
            now = datetime.now()
            print(
                f"[{now}] Waiting for tuning job to complete. Current status: {status}"
            )
            sleep(check_interval)

def delete_file(file_id:str):
    openai.File.delete(file_id)

def wait_file_ready(file_id):
    files = openai.File.list()["data"]
    found = False
    for file in files:
        if file["id"] == file_id:
            found = True
            if file["status"] == "processed":
                return True
            elif file["status"] in ["error", "deleting", "deleted"]:
                print(file)
                raise RuntimeError(
                    f"File upload {file_id} failed with status {file['status']}"
                )
            else:
                return False
    if not found:
        raise RuntimeError(f"File {file_id} not found")
