"""
Convert human judged datasets from "Comparing Inferential Strategies of Humans and Large Language Models in Deductive Reasoning" (Mondorf and Plank, ACL 2024).

Original files located at:
    https://huggingface.co/datasets/mainlp/inferential_strategies
"""

import os
import random
from typing import Any, Literal
from datasets import load_dataset, Dataset

from utils.utils import read_jsonl, save_dict_to_json

DATA_DIR = "original_data"

SCHEMA = {
    "dataset": "Inferential Strategies (Mondorf and Plank, ACL 2024)",
    "dataset_url": "https://huggingface.co/datasets/mainlp/inferential_strategies",
    "annotations": [
        {
            "metric": "Sound Reasoning",
            "category": "categorical",
            "prompt": "{{ instance }}Is the model's reasoning sound, i.e. logically valid? Indicate either 'yes' or 'no'.",
            "labels_list": ["yes", "no"],
        },
    ],
    "expert_annotator": "true",
    "original_prompt": True,
}


def fetch_hf_data(
    dataset_name: str, dataset_config: str, split: str
) -> Dataset:
    """
    Fetches data from a hf-dataset using the provided dataset name, configuration, and split.

    Args:
        dataset_name (str): The name of the dataset to fetch.
        dataset_config (str): The configuration of the dataset.
        split (str, optional): The split of the dataset to fetch.

    Returns:
        Dataset: The fetched dataset.
    """
    return load_dataset(dataset_name, dataset_config, split=split)


def generate_prompt(problem_statement: str, model_response: str) -> str:
    """Generate instance prompt.

    Args:
        problem_statement (str): The problem statement.
        model_response (str): The response of the language model.

    Returns:
        str: The full prompt.
    """
    return (
        "You will be shown the response of a language model to a problem of propositional logic. Your task is to judge whether the model's reasoning is sound, i.e. logically valid.\n"
        "You are first presented with the PROBLEM STATEMENT that the model has been given. Subsequently, the model's RESPONSE is shown. Indicate with a 'yes' if the model's response (in particular its rationale) is sound, and conclude 'no' if it is not.\n\n"
        f"### PROBLEM STATEMENT\n\n{problem_statement}\n\n"
        f"### MODEL RESPONSE\n\n{model_response}\n\n"
    )


def check_sample_alignment(sample1: dict, sample2: dict) -> None:
    """
    Check if two samples have the same alignment by comparing the values of specific keys.

    Args:
        sample1 (dict): The first sample to compare.
        sample2 (dict): The second sample to compare.

    Raises:
        AssertionError: If the values of any of the specified keys do not match.
    """
    for key in ["sample_id", "problem_id", "model_input", "model_reponse", "metadata"]:
        assert (
            sample1[key] == sample2[key]
        ), f"{key} don't match!\nSample1: {sample1[key]}\nSample2: {sample2[key]}"


def find_majority_literal(labels: list[Literal["yes", "no"]]) -> str:
    """
    Find the majority literal ("yes" or "no") in a list of strings.
    In case of a tie, randomly choose between "yes" and "no".

    Parameters:
    labels (list[Literal["yes", "no"]]): List of labels containing "yes" or "no".

    Returns:
    str: The majority literal ("yes" or "no").
    """
    yes_count = labels.count("yes")
    no_count = labels.count("no")

    assert yes_count + no_count == len(labels)

    if yes_count > no_count:
        return "yes"
    elif no_count > yes_count:
        return "no"
    else:
        return random.choice(["yes", "no"])


def create_instance(
    problem_statement: str,
    model_response: str,
    labels: list[Literal["yes", "no"]],
    instance_id: int,
) -> dict[str, Any]:
    """
    Creates an instance dictionary with the given problem statement, model response, labels, and instance ID.

    Parameters:
        problem_statement (str): The problem statement.
        model_response (str): The response from the model.
        labels (list[Literal["yes", "no"]]): The list of labels containing "yes" or "no".
        instance_id (int): The unique ID of the instance.

    Returns:
        dict[str, Any]: A dictionary containing the instance ID, the generated instance, and the annotations.
            The annotations dictionary contains the "Sound Reasoning" key with the majority human label and the individual human scores.
    """
    instance = generate_prompt(
        problem_statement=problem_statement, model_response=model_response
    )

    annotations = {
        "Sound Reasoning": {
            "majority_human": find_majority_literal(labels),
            "individual_human_scores": labels,
        },
    }
    return {"id": instance_id, "instance": instance, "annotations": annotations}


def parse_problem_statement(model_input: str) -> str:
    """
    Parses the problem statement from the given model input.

    Args:
        model_input (str): The input string containing the problem statement.

    Returns:
        str: The parsed problem statement without the instruction section and any trailing characters.
    """
    return model_input.split("### Instruction ###\n")[-1].replace(" [/INST] ", "")


def assemble_instances(
    annotator1_data: Dataset,
    annotator2_data: Dataset,
    instance_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Assembles instances from the given annotator1_data and annotator2_data.

    Args:
        annotator1_data (Dataset): A list of dictionaries representing the data from annotator1.
        annotator2_data (Dataset): A list of dictionaries representing the data from annotator2.
        instance_list (list[dict[str, Any]]): The list of instances to append the assembled instances to.

    Returns:
        list[dict[str, Any]]: The list of assembled instances.

    Raises:
        ValueError: If any of the labels in annotator1_sample or annotator2_sample is not "True" or "False".
    """
    for annotator1_sample, annotator2_sample in zip(annotator1_data, annotator2_data):
        check_sample_alignment(annotator1_sample, annotator2_sample)

        # extract data
        model_response = annotator1_sample["model_reponse"]
        model_input = annotator1_sample["model_input"]
        problem_statement = parse_problem_statement(model_input)

        # labels
        soundness_labels: list[Literal["yes", "no"]] = []
        for label in [
            annotator1_sample["sound_reasoning"],
            annotator2_sample["sound_reasoning"],
        ]:
            if label == "True":
                soundness_labels.append("yes")
            elif label == "False":
                soundness_labels.append("no")
            else:
                raise ValueError(f"Invalid label: {label}!")

        instance = create_instance(
            problem_statement=problem_statement,
            model_response=model_response,
            labels=soundness_labels,
            instance_id=len(instance_list) + 1,
        )
        instance_list.append(instance)

    return instance_list


def convert_data(
    data_path: str,
) -> list[dict[str, Any]]:
    """
    Converts data from a JSONL file to a list of instances.

    Args:
        data_path (str): The path to the JSONL file.

    Returns:
        list[dict[str, Any]]: The list of converted instances.
    """
    instance_list: list[dict[str, Any]] = []

    model_list: list[str] = [
        "llama2_7b_chat_hf",
        "llama2_13b_chat_hf",
        "llama2_70b_chat_hf",
        "mistral_7b_instruct_hf",
        "zephyr_7b_beta_hf",
    ]

    for model in model_list:
        annotator1_data_hf = fetch_hf_data(
            dataset_name="mainlp/inferential_strategies",
            dataset_config="annotator1",
            split=model,
        )

        annotator2_data_hf = fetch_hf_data(
            dataset_name="mainlp/inferential_strategies",
            dataset_config="annotator2",
            split=model,
        )

        instance_list = assemble_instances(
            annotator1_data=annotator1_data_hf,
            annotator2_data=annotator2_data_hf,
            instance_list=instance_list,
        )

    return instance_list


if __name__ == "__main__":
    SCHEMA["instances"] = convert_data(data_path=DATA_DIR)
    save_dict_to_json(SCHEMA, "inferential_strategies.json")
