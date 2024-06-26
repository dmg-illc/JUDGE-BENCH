"""
Convert human judged datasets from "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries" (Wang et al., ACL 2020).

Original files located at:
    https://github.com/W4ngatang/qags/tree/master/data
"""

import os
import random
from typing import Any, Literal

from utils.utils import read_jsonl, save_dict_to_json

DATA_DIR = "original_data"

SCHEMA = {
    "dataset": "qags",
    "dataset_url": "https://github.com/W4ngatang/qags/tree/master/data",
    "annotations": [
        {
            "metric": "Factual Consistency",
            "category": "categorical",
            "prompt": "{{ instance }}Is the sentence factually supported by the article? Indicate either 'yes' or 'no'.",
            "labels_list": ["yes", "no"],
        },
    ],
    "expert_annotator": "false",
    "original_prompt": True,
}


def generate_cnndm_prompt(article: str, summary_sentence: str) -> str:
    """Generate CNN/DM-specific prompt.

    Args:
        article (str): The source article.
        summary_sentence (str): The one-sentence summary to be judged.

    Returns:
        str: The full prompt.
    """
    return (
        "Is the sentence supported by the article?\n\n"
        "In this task, you will read an article and a sentence.\n\n"
        "The task is to determine if the sentence is factually correct given the contents of the article. Many sentences contain portions of text copied directly from the article. "
        "Be careful as some sentences may be combinations of two different parts of the article, resulting in sentences that overall aren't supported by the article. "
        'Some article sentences may seem out of place (for example, "Scroll down for video"). '
        "If the sentence is a copy of an article sentence, including one of these sentences, you should still treat it as factually supported. "
        "Otherwise, if the sentence doesn't make sense, you should mark it as not supported. Also note that the article may be cut off at the end.\n\n"
        f"ARTICLE:\n{article}\n\nSENTENCE:\n{summary_sentence}\n\n"
    )


def generate_xsum_prompt(article: str, summary_sentence: str) -> str:
    """Generate XSUM-specific prompt.

    Args:
        article (str): The source article.
        summary_sentence (str): The one-sentence summary to be judged.

    Returns:
        str: The full prompt.
    """
    return (
        "Is the sentence supported by the article?\n\n"
        "In this task, you will read an article and a sentence.\n\n"
        "The task is to determine if the sentence is factually correct given the contents of the article. "
        "All parts of the sentence must be stated or implied by the article to be considered correctly. "
        'For example, if the sentence discusses "John Smith" but the article only talks about "Mr. Smith", the fact that the person\'s first name is John is NOT supported. '
        "Or, if the sentence mentions a 15-year-old girl but the article only discusses a young girl, the fact that she is 15 is NOT supported. "
        "Verifying a sentence will often require combining facts from many different parts of the article, so read the entire article closely. "
        "If the sentence directly copies the article, you should mark it as supported. If the sentence doesn't make sense, you should mark it as not supported.\n\n"
        f"ARTICLE:\n{article}\n\nSENTENCE:\n{summary_sentence}\n\n"
    )


def generate_prompt(
    dataset: Literal["cnndm", "xsum"],
    article: str,
    sentence: str,
) -> str:
    """
    Generates a prompt based on the given dataset, article, and sentence.

    Args:
        dataset (Literal["cnndm", "xsum"]): The name of the dataset.
        article (str): The source article.
        sentence (str): The one-sentence summary to be judged.

    Returns:
        str: The generated prompt.

    Raises:
        ValueError: If the dataset is invalid.
    """
    if dataset == "cnndm":
        return generate_cnndm_prompt(article=article, summary_sentence=sentence)
    elif dataset == "xsum":
        return generate_xsum_prompt(article=article, summary_sentence=sentence)
    else:
        raise ValueError("Invalid dataset.")


def find_majority_literal(labels: list[str]) -> str:
    """
    Find the majority literal ("yes" or "no") in a list of strings.
    In case of a tie, randomly choose between "yes" and "no".

    Parameters:
    labels (list[str]): List of labels containing "yes" or "no".

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
    dataset: Literal["cnndm", "xsum"],
    article: str,
    sentence: str,
    labels: list[str],
    instance_id: int,
) -> dict[str, Any]:
    """
    Creates an instance dictionary with the given dataset, article, sentence, labels, and instance ID.

    Args:
        dataset (Literal["cnndm", "xsum"]): The name of the dataset.
        article (str): The source article.
        sentence (str): The one-sentence summary to be judged.
        labels (list[str]): The list of labels for the instance.
        instance_id (int): The unique ID of the instance.

    Returns:
        dict: A dictionary containing the instance ID, the generated instance, and the annotations.
            The annotations dictionary contains the "Factual Consistency" key with the majority human label and the individual human scores.
    """
    instance = generate_prompt(dataset=dataset, article=article, sentence=sentence)

    annotations = {
        "Factual Consistency": {
            "majority_human": find_majority_literal(labels),
            "individual_human_scores": labels,
        },
    }
    return {"id": instance_id, "instance": instance, "annotations": annotations}


def assemble_instances(
    data: list[dict],
    dataset: Literal["cnndm", "xsum"],
    instance_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Assembles instances from the given data for the specified dataset.

    Args:
        data (list[dict]): A list of dictionaries representing the data.
        dataset (Literal["cnndm", "xsum"]): The name of the dataset.
        instance_list (list[dict[str, Any]]): The list of instances to append the assembled instances to.

    Returns:
        list[dict[str, Any]]: The list of assembled instances.

    Raises:
        AssertionError: If the number of summary sentences or responses is not 3.

    This function iterates over the given data and assembles instances for each sample.
    It asserts that the number of summary sentences and responses for each sample is 3.
    For each summary sentence, it retrieves the article and responses.
    It then creates an instance using the create_instance function and appends it to the instance_list.
    Finally, it returns the updated instance_list.
    """

    for sample in data:
        summary_sentences = sample["summary_sentences"]
        assert (
            len(summary_sentences) > 0
        ), f"Expected number of summary sentences > 0.\nActual number of summary sentences: {len(summary_sentences)}.\n{summary_sentences}"

        article = sample["article"]
        for sum_dict in summary_sentences:
            sum_sentence = sum_dict["sentence"]
            responses = sum_dict["responses"]
            assert (
                len(responses) == 3
            ), f"Expected number of responses: 3.\nActual number of responses: {len(responses)}.\n{responses}"

            response_labels = [r_dict["response"] for r_dict in responses]
            instance = create_instance(
                dataset=dataset,
                article=article,
                sentence=sum_sentence,
                labels=response_labels,
                instance_id=len(instance_list) + 1,
            )
            instance_list.append(instance)

    return instance_list


def convert_data(
    data_path: str,
    dataset: Literal["cnndm", "xsum"],
    instance_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Converts data from a JSONL file to a list of instances.

    Args:
        data_path (str): The path to the JSONL file.
        dataset (Literal["cnndm", "xsum"]): The name of the dataset.
        instance_list (list[dict[str, Any]]): The list of instances to append the converted instances to.

    Returns:
        list[dict[str, Any]]: The list of converted instances.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        JSONDecodeError: If a line in the JSONL file is not valid JSON.
    """
    data: list[dict] = read_jsonl(data_path)

    return assemble_instances(data=data, dataset=dataset, instance_list=instance_list)


if __name__ == "__main__":
    cnndm_data_path = os.path.join(DATA_DIR, "mturk_cnndm.jsonl")
    xsum_data_path = os.path.join(DATA_DIR, "mturk_xsum.jsonl")

    # cnndm
    cnndm_instances = convert_data(
        data_path=cnndm_data_path, dataset="cnndm", instance_list=[]
    )

    # xsum
    all_instances = convert_data(
        data_path=xsum_data_path, dataset="xsum", instance_list=cnndm_instances
    )

    # add instances to schema
    SCHEMA["instances"] = all_instances

    # save to dics
    save_dict_to_json(SCHEMA, "qags.json")
