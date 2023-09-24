import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

from args import parse_args, print_args, TaskArguments, InstanceArguments


def create_sagemaker_role():
    # use boto3 to create a sagemaker role with full access to s3
    iam = boto3.client("iam")
    role = iam.create_role(
        RoleName="sagemaker_execution_role",
        AssumeRolePolicyDocument="""{
            "Version": "2012-10-17",
            "Statement": [
                {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
                }
            ]
            }""",
    )

    iam.attach_role_policy(
        RoleName=role["Role"]["RoleName"],
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )

    iam.attach_role_policy(
        RoleName=role["Role"]["RoleName"],
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
    )

    return role


def get_sagemaker_role_arn():
    # use boto3 to create a sagemaker role with full access to s3
    iam = boto3.client("iam")
    try:
        role = iam.get_role(RoleName="sagemaker_execution_role")
    except iam.exceptions.NoSuchEntityException:
        role = create_sagemaker_role()

    return role["Role"]["Arn"]


metric_definitions = [
    {"Name": "loss", "Regex": "'loss': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "learning_rate", "Regex": "'learning_rate': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_loss", "Regex": "'eval_loss': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_accuracy", "Regex": "'eval_accuracy': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_f1", "Regex": "'eval_f1': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_precision", "Regex": "'eval_precision': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_recall", "Regex": "'eval_recall': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_runtime", "Regex": "'eval_runtime': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_samples_per_second", "Regex": "'eval_samples_per_second': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "eval_steps_per_second", "Regex": "'eval_steps_per_second': ([0-9]+(.|e\-)[0-9]+),?"},
    {"Name": "epoch", "Regex": "'epoch': ([0-9]+(.|e\-)[0-9]+),?"},
]


def main():
    args = parse_args(TaskArguments, InstanceArguments)

    print_args(args)

    sess = sagemaker.Session()

    sagemaker_session_bucket = None
    if sagemaker_session_bucket is None and sess is not None:
        sagemaker_session_bucket = sess.default_bucket()

    role = get_sagemaker_role_arn()

    environment = {
        "HUGGING_FACE_HUB_TOKEN": os.environ["HUGGING_FACE_HUB_TOKEN"],
        "HUGGINGFACE_HUB_CACHE": "/tmp/.cache",
    }

    # create the Estimator
    huggingface_estimator = HuggingFace(
        entry_point="launch.py",  # train script
        source_dir="./",  # directory which includes all the files needed for training
        base_job_name=args.task.project,  # the name of the training job
        role=role,  # Iam role used in training job to access AWS ressources, e.g. S3
        transformers_version="4.28",  # the transformers version used in the training job
        pytorch_version="2.0",  # the pytorch_version version used in the training job
        py_version="py310",  # the python version used in the training job
        hyperparameters={
            "output_dir": "/opt/ml/model",  # pass output_dir to launch.py
            "save_dir": "/opt/ml/model",  # pass save_dir to launch.py
            "tmp_dir": "/tmp",  # pass tmp_dir to launch.py
        },
        environment=environment,  # set env variable to cache models in /tmp
        disable_output_compression=True,  # not compress output to save training time and cost
        metric_definitions=metric_definitions,  # report metrics for hyperparameter tuning
        **args.instance.config(),
    )

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(wait=True)


if __name__ == "__main__":
    main()
