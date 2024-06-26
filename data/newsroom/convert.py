"""
Convert human judged datasets from "Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies" (Grusky et al., NAACL 2018).

Original files located at:
    https://github.com/lil-lab/newsroom/tree/master/humaneval
"""

import os
import html
from typing import Any
import statistics
from collections import Counter
from pandas import DataFrame

from utils.utils import read_csv_to_dataframe, save_dict_to_json

DATA_DIR = "original_data"

SCHEMA = {
    "dataset": "newsroom",
    "dataset_url": "https://github.com/lil-lab/newsroom/tree/master/humaneval",
    "annotations": [
        {
            "metric": "Informativeness",
            "category": "graded",
            "prompt": "On a scale of 1 (low) to 5 (high), how well does the summary capture the key points of the article?\n\n{{ instance }}",
            "worst": 1,
            "best": 5,
        },
        {
            "metric": "Relevance",
            "category": "graded",
            "prompt": "On a scale of 1 (low) to 5 (high), are the details provided by the summary consistent with details in the article?\n\n{{ instance }}",
            "worst": 1,
            "best": 5,
        },
        {
            "metric": "Fluency",
            "category": "graded",
            "prompt": "On a scale of 1 (low) to 5 (high), are the individual sentences of the summary well-written and grammatical?\n\n{{ instance }}",
            "worst": 1,
            "best": 5,
        },
        {
            "metric": "Coherence",
            "category": "graded",
            "prompt": "On a scale of 1 (low) to 5 (high), do phrases and sentences of the summary fit together and make sense collectively?\n\n{{ instance }}",
            "worst": 1,
            "best": 5,
        },
    ],
    "expert_annotator": "false",
    "original_prompt": True,
}


def generate_prompt(summary: str, title: str, article: str) -> str:
    """Generate instance prompt.

    Args:
        summary (str): The system summary to be judged.
        title (str): The title of the source article.
        article (str): The source article.

    Returns:
        str: The full prompt.
    """
    return (
        f"### Generated Summary\n\n{summary}\n\n"
        f"### Source Article\n\n{title}\n{article}"
    )


def compute_majority_vote(numbers: list[int]) -> int | None:
    """
    Computes the majority vote of a list of integers.

    Args:
        numbers (list[int]): A list of integers.

    Returns:
        int | None: The majority vote. None if no majority rating can be found.
    """
    if not numbers:
        raise ValueError("The list of numbers is empty.")

    count = Counter(numbers)
    majority_vote, majority_count = count.most_common(1)[0]

    # Check if the majority vote is indeed a majority
    if majority_count > len(numbers) / 2:
        return majority_vote
    else:
        return None


def create_instance(df_instance: DataFrame, instance_id: int) -> dict[str, Any]:
    """
    Creates an instance dictionary based on the given DataFrame and instance ID.

    Args:
        df_instance (DataFrame): The DataFrame containing the instance data.
        instance_id (int): The unique ID of the instance.

    Returns:
        dict: A dictionary representing the instance
    """
    # check that instance has same 'ArticleText', 'SystemSummary', and 'ArticleTitle' across entries
    assert (
        df_instance["ArticleText"].nunique() == 1
    ), f"'ArticleText' is inconsistent across rows!\n{df_instance}"
    assert (
        df_instance["SystemSummary"].nunique() == 1
    ), f"'SystemSummary' is inconsistent across rows!\n{df_instance}"
    assert (
        df_instance["ArticleTitle"].nunique() == 1
    ), f"'ArticleTitle' is inconsistent across rows!\n{df_instance}"

    summary = html.unescape(df_instance["SystemSummary"].iloc[0])
    title = html.unescape(df_instance["ArticleTitle"].iloc[0])
    article = html.unescape(df_instance["ArticleText"].iloc[0])

    instance_prompt = generate_prompt(summary=summary, title=title, article=article)

    informativeness_rating: list[int] = [
        row["InformativenessRating"] for _, row in df_instance.iterrows()
    ]
    relevance_rating: list[int] = [
        row["RelevanceRating"] for _, row in df_instance.iterrows()
    ]
    fluency_rating: list[int] = [
        row["FluencyRating"] for _, row in df_instance.iterrows()
    ]
    coherence_rating: list[int] = [
        row["CoherenceRating"] for _, row in df_instance.iterrows()
    ]

    annotations = {
        "Informativeness": {
            "mean_human": round(statistics.mean(informativeness_rating), 2),
            "majority_human": compute_majority_vote(informativeness_rating),
            "individual_human_scores": informativeness_rating,
        },
        "Relevance": {
            "mean_human": round(statistics.mean(relevance_rating), 2),
            "majority_human": compute_majority_vote(relevance_rating),
            "individual_human_scores": relevance_rating,
        },
        "Fluency": {
            "mean_human": round(statistics.mean(fluency_rating), 2),
            "majority_human": compute_majority_vote(fluency_rating),
            "individual_human_scores": fluency_rating,
        },
        "Coherence": {
            "mean_human": round(statistics.mean(coherence_rating), 2),
            "majority_human": compute_majority_vote(coherence_rating),
            "individual_human_scores": coherence_rating,
        },
    }

    # remove invalid majority votings
    for _, metric_dict in annotations.items():
        if metric_dict["majority_human"] is None:
            del metric_dict["majority_human"]

    instance = {
        "id": instance_id,
        "instance": instance_prompt,
        "annotations": annotations,
    }

    return instance


def assemble_instances(
    data_frame: DataFrame,
) -> list[dict[str, Any]]:
    """
    Assembles instances from the given DataFrame by grouping it based on "ArticleID" and "System" columns.

    Args:
        data_frame (DataFrame): The DataFrame containing the data to assemble instances from.

    Returns:
        list[dict[str, Any]]: A list of dictionaries representing the assembled instances.
    """
    instance_list: list[dict[str, Any]] = []
    data_instances = data_frame.groupby(["ArticleID", "System"])

    for (article_id, system), group in data_instances:
        assert (
            len(group)
        ) == 3, f"3 annotations for each article-system pair expected. However, got:\n{group}"
        instance = create_instance(group, len(instance_list) + 1)
        instance_list.append(instance)

    return instance_list


def convert_data(
    data_path: str,
) -> None:
    """
    Convert data from a CSV file to a list of instances and save it to a JSON file.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        None
    """
    data_frame = read_csv_to_dataframe(data_path)

    # get instances
    instances = assemble_instances(data_frame)
    SCHEMA["instances"] = instances

    # save converted data
    save_dict_to_json(SCHEMA, "newsroom.json")


if __name__ == "__main__":
    convert_data(data_path=os.path.join(DATA_DIR, "newsroom-human-eval.csv"))
