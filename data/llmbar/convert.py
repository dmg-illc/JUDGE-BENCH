# -*- coding: utf-8 -*-
"""

Convert LLMBar dataset (Zeng et al., ICLR 2024).

Original files are located at
    https://github.com/princeton-nlp/LLMBar
    https://huggingface.co/datasets/princeton-nlp/LLMBar

"""

from datasets import load_dataset

from llm_metaeval.data.dataclasses import (
    CategoricalAnnotation,
    CategoricalAnnotationScores,
    Dataset,
    Instance,
)

# Load dataset from Hugging Face dataset hub
llmbar = load_dataset("princeton-nlp/LLMBar", "LLMBar")


# No human guidelines are available for this dataset, as instances were constructed to be right or wrong by design.
# We will use the original "Vanilla" prompt as the human guideline, but with system-specific tokens removed.
# More prompts available at `data/llmbar/prompts`, see `data/llmbar/prompts/README.md` for more information
with open('prompts/comparison/Vanilla_Model_Agnostic.txt', 'r') as file:
    vanilla_prompt = file.read()
vanilla_prompt = vanilla_prompt.replace('{input}', '{{ input }}')
vanilla_prompt = vanilla_prompt.replace('{output_1}', '{{ output_a }}')
vanilla_prompt = vanilla_prompt.replace('{output_2}', '{{ output_b }}')


# Transform label from integer to string
def transform_label(label):
    if label == 1:
        return "model_a"
    elif label == 2:
        return "model_b"
    else:
        raise ValueError("Invalid label")


# Collect instances from the "Natural" split
natural_instances = []
for idx, entry in enumerate(llmbar["Natural"]):
    instance = Instance(
        id=f"Natural_{idx}",
        instance={
            "input": entry["input"],
            "output_a": entry["output_1"],
            "output_b": entry["output_2"]
        },
        annotations={
            "quality": CategoricalAnnotationScores(
                majority_human=transform_label(entry["label"]),
                individual_human_scores=[transform_label(entry["label"])]
            )
        },
    )
    natural_instances.append(instance)

# Create and dump the "Natural" dataset
dataset = Dataset(
    dataset="LLMBar Natural (Zeng et al., ICLR 2024)",
    dataset_url="https://github.com/princeton-nlp/LLMBar",
    expert_annotator="true",
    original_prompt=True,
    annotations=[
        CategoricalAnnotation(
            metric="quality_single_turn",
            category="categorical",
            prompt=vanilla_prompt,
            labels_list=["model_a", "model_b"],
        )
    ],
    instances=natural_instances
)
with open("data-natural.json", "w") as file:
    file.write(dataset.model_dump_json(indent=4))


# Collect instances from the adversarial splits
adversarial_instances = []
for adversarisal_split in ["Adversarial_Neighbor", "Adversarial_GPTInst",  "Adversarial_GPTOut", "Adversarial_Manual"]: 

    for idx, entry in enumerate(llmbar[adversarisal_split]):
        instance = Instance(
            id=f"{adversarisal_split}_{idx}",
            instance={
                "input": entry["input"],
                "output_a": entry["output_1"],
                "output_b": entry["output_2"]
            },
            annotations={
                "quality": CategoricalAnnotationScores(
                    majority_human=transform_label(entry["label"]),
                    individual_human_scores=[transform_label(entry["label"])]
                )
            },
        )
        adversarial_instances.append(instance)

# Create and dump the "Adversarial" dataset
dataset = Dataset(
    dataset="LLMBar Adversarial (Zeng et al., ICLR 2024)",
    dataset_url="https://github.com/princeton-nlp/LLMBar",
    expert_annotator="true",
    original_prompt=True,
    annotations=[
        CategoricalAnnotation(
            metric="quality_single_turn",
            category="categorical",
            prompt=vanilla_prompt,
            labels_list=["model_a", "model_b"],
        )
    ],
    instances=adversarial_instances
)
with open("data-adversarial.json", "w") as file:
    file.write(dataset.model_dump_json(indent=4))
