from typing import Optional, Callable
from time import sleep
from datetime import datetime
import os


def create_tuning_job(
    client: Callable,
    model: str,
    training_file: str,
    n_epochs: Optional[str] = None,
    suffix: Optional[str] = None,
):
    out = client.files.create(file=open(training_file, "rb"), purpose="fine-tune")
    out_id = out.id
    print(f"Created new file. FILE_ID = {out_id}")
    print(f"Waiting for file to be processed ...")
    while not wait_file_ready(client, out_id):
        sleep(5)
    # delete the training_file after it is uploaded
    os.remove(training_file)
    params = {
        "model": model,
        "training_file": out_id,
    }
    if n_epochs is not None:
        params["hyperparameters"] = {"n_epochs": n_epochs}
    if suffix is not None:
        params["suffix"] = suffix
    return client.fine_tuning.jobs.create(**params)


def await_results(client: Callable, job_id: str, check_interval: int = 120):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
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


def delete_file(client: Callable, file_id: str):
    client.files.delete(file_id)


def wait_file_ready(client: Callable, file_id):
    files = client.files.list().data
    found = False
    for file in files:
        if file.id == file_id:
            found = True
            if file.status == "processed":
                return True
            elif file.status in ["error", "deleting", "deleted"]:
                print(file)
                raise RuntimeError(
                    f"File upload {file_id} failed with status {file.status}"
                )
            else:
                return False
    if not found:
        raise RuntimeError(f"File {file_id} not found")
