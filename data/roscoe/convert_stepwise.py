"""
Convert human judged datasets from "ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning" (Golovneva et al., ICLR 2023).
Only consider metrics that consider individual reasoning steps of the model's generated answer.

Original files located at:
    https://dl.fbaipublicfiles.com/parlai/projects/roscoe/annotations.zip
"""

import os
import pandas as pd
from typing import Any

from utils.utils import (
    read_csv,
    read_jsonl,
    parse_reasoning_chain,
    compare_strings,
    split_substeps,
    save_dict_to_json,
)


DATASET_NAMES = [
    "drop",
    "cosmos",
    "esnli",
    "gsm8k",
]

SCHEMA = {
    "dataset_url": "https://dl.fbaipublicfiles.com/parlai/projects/roscoe/annotations.zip",
    "annotations": [
        {
            "metric": "Grammar",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any faulty, unconventional, or controversial grammar usage? In other words, does the language in this step sounds unnatural?",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Factuality",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any information that contradicts the context while still largely talking about the same concepts? (Ex. Characteristics of named objects are wrong, named entities changed.)",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Coherency and Logic",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any logical deduction errors (Ie, makes a conclusion contradictory to previously stated clauses, including clauses within this step itself; makes a conclusion while not having enough support to make the conclusion)",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Final Answer",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain a final step with an incorrect final answer? (If an explicit 'yes/no' is not provided, an exact match of the correct answer with respect to the question in the context must be given.)",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Hallucination",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any information not provided in the problem statement that is irrelevant or wrong?",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Redundancy",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any information not required to answer the question asked despite being factual and consistent with the context?",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Repetition",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any information, possibly paraphrased, already mentioned in previous step (and thus could be dropped without impacting correctness)?",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Commonsense",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any errors in relation to general knowledge about the world (i.e. how to compute velocity, how many inches in one foot, etc) not explicitly provided in the context?",
            "labels_list": ["yes", "no"],
        },
        {
            "metric": "Arithmetic",
            "category": "categorical",
            "prompt": "{{ instance }}Does this step contain any math equation errors? Note that you should consider only current step in isolation, rather than issues propagated from prior steps.",
            "labels_list": ["yes", "no"],
        },
    ],
    "expert_annotator": "true",
    "original_prompt": True,
}


def generate_prompt(
    premise: str, hypothesis: str, correct_relationship: str, generated_response: str
) -> str:
    """Generate prompt for datasets other than gsm8k.

    Args:
        premise (str): The premise of the problem statement.
        hypothesis (str): The hypothesis of the problem statement.
        correct_relationship (str): The correct relationship of the problem statement.
        generated_response (str): The model's generated response.

    Returns:
        str: The full prompt.
    """
    return (
        'For this task, you will be shown a CONTEXT with a "Situation" and a "Claim" about that "Situation". The "Claim" may or may not be supported by the "Situation". The Correct Relationship between the "Claim" and the "Situation" is provided.\n\n'
        "You will be shown a GENERATED RESPONSE generated from a bot, asked the question\n\nIs the Claim supported by the Situation?\n\nYou will be asked to judge the individual STEPS within the GENERATED RESPONSE. Interpret the questions to the best of your ability. "
        'Sometimes the generated response will refer to the "Situation" as a "Premise" and the "Claim" as a "Hypothesis". It will oftentimes be faster to read the "Claim" before the "Situation".\n\n'
        f"CONTEXT:\nSituation (Premise): {premise}\n\nClaim (Hypothesis): {hypothesis}\n\nIs the Claim supported by the Situation?\n\nCorrect Relationship (Yes or No): {correct_relationship}\n\nGENERATED RESPONSE:\n{generated_response}\n"
    )


def generate_gsm8k_prompt(
    question: str, correct_answer: str, generated_response: str
) -> str:
    """Generate gsm8k-specific prompt.

    Args:
        question (str): The problem statement.
        correct_answer (str): The ground truth solution to the problem statement.
        generated_response (str): The model's generated response.

    Returns:
        str: The full prompt.
    """
    return (
        'For this task, you will be shown a CONTEXT with a "Question" and a corresponding "Solution".\n\n'
        'You will be shown a GENERATED RESPONSE generated from a bot, asked to solve the "Question".\n\nYou will be asked to judge the individual STEPS within the GENERATED RESPONSE. Interpret the questions to the best of your ability.\n\n'
        f"CONTEXT:\nQuestion: {question}\n\Solution: {correct_answer}\n\nGENERATED RESPONSE:\n{generated_response}\n"
    )


def create_input_prompt(
    dataset_name: str, context: dict[str, str], generated_response: str
) -> str:
    """
    Generates an input prompt based on the dataset name.

    Args:
        dataset_name (str): The name of the dataset.
        context (dict[str, str]): A dictionary containing context information.
        generated_response (str): The model's generated response.

    Returns:
        str: The input prompt generated based on the dataset name.
    """
    if dataset_name == "gsm8k":
        return generate_gsm8k_prompt(
            question=context["premise"].strip(),
            correct_answer=context["hypothesis"].split(
                "IGNORE THIS. Ground truth here for reference. "
            )[-1],
            generated_response=generated_response,
        )
    else:
        return generate_prompt(
            premise=context["premise"].strip(),
            hypothesis=context["hypothesis"].strip(),
            correct_relationship=context["answer"].strip(),
            generated_response=generated_response,
        )


def create_instance(
    id: int, step_id: int, context_prompt: str, annotation_data: pd.Series
) -> dict[str, Any]:
    """
    A function to create an instance with annotations for overall quality, coherency, missing steps, and contradiction based on the provided parameters.

    Args:
        id (int): The ID of the instance.
        context_prompt (str): The context prompt for the instance.
        annotation_data (pd.Series): The DataSeries containing annotation data.

    Returns:
        dict[str, Any]: A dictionary representing the instance with annotations.
    """
    reasoning_step = annotation_data[f"{step_id}_step_text"]
    instruction = f"JUDGE: {reasoning_step}\n"

    # assemble annotations
    grammar = annotation_data[f"{step_id}_step_step_questions_newGrammar_result"]
    factuality = annotation_data[
        f"{step_id}_step_step_questions_newContradictContext_result"
    ]
    logic = annotation_data[f"{step_id}_step_step_questions_newLogicalDeduction_result"]
    final_answer = annotation_data[
        f"{step_id}_step_step_questions_newFinalAnswerWrong_result"
    ]
    hallucination = annotation_data[
        f"{step_id}_step_step_questions_newExtraUselessInfo_result"
    ]
    redundancy = annotation_data[
        f"{step_id}_step_step_questions_newIntermediateFactualInfo_result"
    ]
    repetition = annotation_data[
        f"{step_id}_step_step_questions_newDroppableStep_result"
    ]
    commonsense = annotation_data[
        f"{step_id}_step_step_questions_newWorldKnowledge_result"
    ]
    arithmetic = annotation_data[f"{step_id}_step_step_questions_newMathError_result"]

    annotations = {
        "Grammar": {
            "majority_human": grammar,
            "individual_human_scores": [grammar],
        },
        "Factuality": {
            "majority_human": factuality,
            "individual_human_scores": [factuality],
        },
        "Coherency and Logic": {
            "majority_human": logic,
            "individual_human_scores": [logic],
        },
        "Final Answer": {
            "majority_human": final_answer,
            "individual_human_scores": [final_answer],
        },
        "Hallucination": {
            "majority_human": hallucination,
            "individual_human_scores": [hallucination],
        },
        "Redundancy": {
            "majority_human": redundancy,
            "individual_human_scores": [redundancy],
        },
        "Repetition": {
            "majority_human": repetition,
            "individual_human_scores": [repetition],
        },
        "Commonsense": {
            "majority_human": commonsense,
            "individual_human_scores": [commonsense],
        },
        "Arithmetic": {
            "majority_human": arithmetic,
            "individual_human_scores": [arithmetic],
        },
    }

    instance = {
        "id": id,
        "instance": context_prompt + instruction,
        "annotations": annotations,
    }

    return instance


def assemble_instances(
    context_file: str, annotation_file: str, dataset_name: str
) -> list[dict]:
    """
    Assembles instances of the dataset.

    Args:
        context_file (str): The file containing context data.
        annotation_file (str): The file containing annotation data.
        dataset_name (str): The name of the dataset.

    Returns:
        list[dict]: A list of dictionaries representing the assembled instances.
    """
    instances_stepwise: list[dict] = []

    # read data
    context_data = read_jsonl(context_file)
    annotation_data = read_csv(annotation_file)

    for idx, annotation_row in annotation_data.iterrows():
        meta_data_idx = annotation_row["metadata_example_idx"]
        context = context_data[meta_data_idx]

        # reasoning chain
        reasoning_chain_context = context["gpt-3"].strip()
        reasoning_chain_annotations = parse_reasoning_chain(
            annotation_row["metadata_generation"]
        ).strip()

        assert compare_strings(
            reasoning_chain_context, reasoning_chain_annotations
        ), f"Annotation reasoning chain for idx {idx} does not match with context!\n{reasoning_chain_context}\n\n{reasoning_chain_annotations}"

        # assemble instance
        reasoning_steps = split_substeps(annotation_row["metadata_generation"])
        generated_response = "\n".join(reasoning_steps)

        context_prompt = create_input_prompt(
            dataset_name=dataset_name,
            context=context,
            generated_response=generated_response,
        )

        # get instances for each step
        for step_id, _ in enumerate(reasoning_steps):
            overall_instance = create_instance(
                id=len(instances_stepwise) + 1,
                step_id=step_id + 1,
                context_prompt=context_prompt,
                annotation_data=annotation_row,
            )
            instances_stepwise.append(overall_instance)

    return instances_stepwise


if __name__ == "__main__":
    context_data_path = os.path.join("original_data", "context")
    annotation_data_path = os.path.join("original_data", "annotated")

    for dataset_name in DATASET_NAMES:
        converted_dataset_name = f"roscoe-{dataset_name}-stepwise"
        context_file = os.path.join(context_data_path, f"{dataset_name}.jsonl")
        annotation_file = os.path.join(annotation_data_path, f"{dataset_name}.csv")

        instances = assemble_instances(context_file, annotation_file, dataset_name)

        # assemble dataset
        dataset_schema: dict[str, Any] = {
            "dataset": f"{converted_dataset_name} (Golovneva et al., ICLR 2023)"
        }
        dataset_schema.update(SCHEMA)
        dataset_schema["instances"] = instances

        # save to dics
        save_dict_to_json(dataset_schema, f"{converted_dataset_name}.json")
