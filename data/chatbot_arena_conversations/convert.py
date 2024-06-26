from datasets import load_dataset
from tqdm import tqdm

import sys

sys.path.append("../..")

from llm_metaeval.data.dataclasses import (
    CategoricalAnnotation,
    CategoricalAnnotationScores,
    Dataset,
    Instance,
)

# Prompt derived from: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl
SYSTEM_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI
assistants to the user question displayed below. You should choose the assistant that follows the
user's instructions and answers the user's question better. Your evaluation should consider factors
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their
responses. Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented does not
influence your decision. Do not allow the length of the responses to influence your evaluation. Do
not favor certain names of the assistants. Be as objective as possible. After providing your
explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A
is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

# Conversation with model A
{{ conversation_a }}

# Conversation with model B
{{ conversation_b }}
"""


def format_conversation(conversation):
    conversation_str = ""

    for idx, turn in enumerate(conversation):
        if turn["role"] == "user":
            conversation_str += f"User: {turn['content']}\n"
        else:
            conversation_str += f"Assistant: {turn['content']}\n"

    return conversation_str


def main():
    hf_dataset = load_dataset("lmsys/chatbot_arena_conversations")

    instances = []

    for idx, ex in enumerate(tqdm(hf_dataset["train"])):
        conversation_a = format_conversation(ex["conversation_a"])
        conversation_b = format_conversation(ex["conversation_b"])

        instance = Instance(
            id=idx,
            instance={
                "conversation_a": conversation_a,
                "conversation_b": conversation_b,
            },
            annotations={
                "quality_single_turn": CategoricalAnnotationScores(
                    majority_human=ex["winner"],
                    individual_human_scores=[ex["winner"]],
                )
            },
        )

        instances.append(instance)

    dataset = Dataset(
        dataset="lmsys/chatbot_arena_conversations",
        dataset_url="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations",
        expert_annotator="false",
        original_prompt=True,
        annotations=[
            CategoricalAnnotation(
                metric="quality_single_turn",
                category="categorical",
                prompt=SYSTEM_PROMPT,
                labels_list=["model_a", "model_b", "tie", "tie (bothbad)"],
            )
        ],
        instances=instances,
    )

    with open("data.json", "w") as file:
        file.write(dataset.model_dump_json())


if __name__ == "__main__":
    main()
