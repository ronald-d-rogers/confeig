import time
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

from args import parse_instance_args


def main():
    task, inst = parse_instance_args()

    sess = sagemaker.Session()

    sagemaker_session_bucket = None
    if sagemaker_session_bucket is None and sess is not None:
        sagemaker_session_bucket = sess.default_bucket()

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    # define Training Job Name
    job_name = f'{task.project}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

    # create the Estimator
    huggingface_estimator = HuggingFace(
        entry_point="launch.py",  # train script
        source_dir="./",  # directory which includes all the files needed for training
        base_job_name=job_name,  # the name of the training job
        role=role,  # Iam role used in training job to access AWS ressources, e.g. S3
        transformers_version="4.28",  # the transformers version used in the training job
        pytorch_version="2.0",  # the pytorch_version version used in the training job
        py_version="py310",  # the python version used in the training job
        hyperparameters={
            "output_dir": "/opt/ml/model",  # pass output_dir to launch.py
            "save_dir": "/opt/ml/model",  # pass save_dir to launch.py
            "tmp_dir": "/tmp",  # pass tmp_dir to launch.py
        },
        environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"},  # set env variable to cache models in /tmp
        disable_output_compression=True,  # not compress output to save training time and cost
        **inst.config(),
    )

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(wait=True)


if __name__ == "__main__":
    main()
