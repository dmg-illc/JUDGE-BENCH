import argparse
from datetime import datetime
import json
from huggingface_hub import login
import os
from models import HFModel, APIModel

dataset_names = {
    "cola": "cola",
    "cola-grammar": "cola-grammar",
    "dailydialog-acceptability": "data",
    "inferential-strategies": "inferential_strategies",
    "llmbar-adversarial": "data-adversarial",
    "llmbar-natural": "data-natural",
    "medical-safety": "data",
    "newsroom": "newsroom",
    "persona_chat": "persona_chat_short",
    "qags": "qags",
    "recipe_crowd_sourcing_data": "meta_evaluation_recipes",
    "roscoe-cosmos": "roscoe-cosmos-overall",
    "roscoe-drop": "roscoe-drop-overall",
    "roscoe-esnli": "roscoe-esnli-overall",
    "roscoe-gsm8k": "roscoe-gsm8k-overall",
    "summeval": "summeval",
    "switchboard-acceptability": "data",
    "topical_chat": "topical_chat_short",
    "toxic_chat-train": "toxic_chat_train",
    "toxic_chat-test": "toxic_chat_test",
    "wmt-human_en_de": "wmt-human_en_de",
    "wmt-human_zh_en": "wmt-human_zh_en",
    "wmt-23_en_de": "wmt-23_en_de",
    "wmt-23_zh_en": "wmt-23_zh_en",
    "dices_990": "dices_990",
    "dices_350_expert": "dices_350_expert",
    "dices_350_crowdsourced": "dices_350_crowdsourced",
}

double_names = {
    "llmbar-adversarial": "llmbar",
    "llmbar-natural": "llmbar",
    "roscoe-cosmos": "roscoe",
    "roscoe-drop": "roscoe",
    "roscoe-esnli": "roscoe",
    "roscoe-gsm8k": "roscoe",
    "toxic_chat-train": "toxic_chat",
    "toxic_chat-test": "toxic_chat",
    "wmt-human_en_de": "wmt-human",
    "wmt-human_zh_en": "wmt-human",
    "wmt-23_en_de": "wmt-23",
    "wmt-23_zh_en": "wmt-23",
    "dices_990": "dices",
    "dices_350_expert": "dices",
    "dices_350_crowdsourced": "dices",
}

lower_batch_size = [
    "llmbar-adversarial",
    "medical-safety",
    "qags",
    "roscoe-cosmos",
    "roscoe-drop",
    "roscoe-esnli",
    "roscoe-gsm8k",
    "summeval",
    "topical_chat",
    "wmt-human",
]

lowest_batch_size = [
    "chatbot_arena_conversations",
    "inferential-strategies",
    "newsroom",
]

api_models = [
    "gpt-4o",
    "gpt-3.5-turbo-0125",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.0-pro-latest",
    "gemini-1.0-pro",
]

additional_prompts = {
    0: (
        "",
        " Do not output anything else. Do not explain or justify your answer. Please output only one answer, and nothing else. Answer with one of {}.",
    ),
    1: ("", " Answer with one of {}. Do not explain your answer."),
    2: ("", " Label: "),
    3: (
        """You should first provide a brief explanation of your evaluation, and then always end your response with either {therefore} verbatim.
Do NOT say all / neither are good.
Do NOT output any other words.
Do NOT say {correct} at the beginning. You should do reasoning and thinking **before** claiming which is the correct answer.
# Instruction:
""",
        """
# Decision (Give a brief explanation of your evaluation followed by either {therefore} verbatim. Always claim which is the correct answer at the end. In your explanation, you should always use one of {} to refer to the answers.):""",
    ),
    4: (
        """# Instruction:
""",
        """
# Decision (Give a brief explanation of your evaluation followed by either {therefore} verbatim. Always claim which is the correct answer at the end. In your explanation, you should always use one of {} to refer to the answers.):""",
    ),
    5: (
        "",
        " Answer with a number between 0 and 100. Do not explain your answer.",
    ),
}

system_prompts = {
    0: "You are a helpful but laconic assistant, skilled in answering language questions very briefly.",
    1: "Answer with one of {}. Do not explain your answer. Do not output anything else. Format your answer as 'Label: <value>'.",
    2: """You should first provide a brief explanation of your evaluation, and then always end your response with either {therefore} verbatim.
Do NOT say all / neither are good.
Do NOT output any other words.
Do NOT say {correct} at the beginning. You should do reasoning and thinking **before** claiming which is the correct answer.
""",
}


def replace_instance(prompt, instance):
    initial_prompt = prompt
    if type(instance["instance"]) == str:
        prompt = prompt.replace("{{ instance }}", instance["instance"])
        if prompt == initial_prompt:
            print(prompt)
            raise Exception("Prompt was incorrectly processed")
        return prompt
    elif type(instance["instance"]) == dict:
        for part in instance["instance"]:
            if not instance["instance"][part]:
                instance["instance"][part] = ""
            prompt = prompt.replace(
                "{{ " + part + " }}", instance["instance"][part]
            )
        if prompt == initial_prompt:
            print(prompt)
            raise Exception("Prompt was incorrectly processed")
        return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="all",
        help="Select only one specific dataset",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        default=None,
        help="Huggingface token that grants access to HuggingFace models",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size used for generating model responses",
    )
    parser.add_argument(
        "-nt",
        "--new_tokens",
        type=int,
        default=None,
        help="Number of tokens that model completion is limited to",
    )
    parser.add_argument(
        "-ap",
        "--add_prompt",
        type=int,
        default=0,
        help="Additional (force) prompt to add to dataset prompt",
    )

    parser.add_argument(
        "-sp",
        "--system_prompt",
        type=int,
        default=None,
        help="The index of the system prompt to use, if any",
    )

    parser.add_argument(
        "-rd",
        "--results_dir",
        type=str,
        default="results",
        help="Directory for storing results",
    )
    args = parser.parse_args()

    system_prompt = (
        system_prompts[args.system_prompt]
        if args.system_prompt is not None
        else None
    )
    # create a Model class instance, setting batch size and number of generated tokens
    if args.model in api_models:
        model = APIModel(args.model, new_tokens=args.new_tokens)
    else:
        if args.token is not None:
            login(args.token)
        model = HFModel(args.model, new_tokens=args.new_tokens)

    for dataset_name, filename in dataset_names.items():
        if args.dataset != "all" and dataset_name != args.dataset:
            continue
        print(dataset_name)

        resolved_name = (
            double_names[dataset_name]
            if dataset_name in double_names
            else dataset_name
        )
        filepath = f"data/{resolved_name}/{filename}.json"
        print(filepath)

        # load dataset
        with open(filepath, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        # set batch size
        if dataset_name in lower_batch_size:
            batch_size = max(1, args.batch_size // 2)
        elif dataset_name in lowest_batch_size:
            batch_size = max(1, args.batch_size // 8)
        else:
            batch_size = args.batch_size
        # save run details
        data["run_details"] = {
            "additional_prompt_id": args.add_prompt,
            "additional_prompt": additional_prompts[args.add_prompt],
            "system_prompt_id": args.system_prompt,
            "system_prompt": system_prompt,
            "model_specific_prompt": False,
            "few-shot": False,
            "fp16": True,
            "model": args.model,
            "batch_size": batch_size,
            "n_new_tokens": args.new_tokens,
        }

        # get list of labels
        def get_label_list(annotation):
            """
            Given an annotation, returns the list of corresponding labels
            """
            if "labels_list" in annotation:
                labels = annotation["labels_list"]
            else:
                lowest_label = min(annotation["best"], annotation["worst"])
                highest_label = max(annotation["best"], annotation["worst"])
                if type(lowest_label) != int:
                    labels = [str(lowest_label), str(highest_label)]
                else:
                    labels = [
                        str(i) for i in range(lowest_label, highest_label + 1)
                    ]
            return {
                "labels_only": ", ".join(labels),
                "therefore": '"Therefore, '
                + ' is correct." or "Therefore, '.join(labels)
                + ' is correct."',
                "correct": '"'
                + ' is correct." or "'.join(labels)
                + ' is correct."',
            }

        label_lists = {
            annotation["metric"]: get_label_list(annotation)
            for annotation in data["annotations"]
        }
        # get prompt for each metric with correct labels
        prompts = {
            annotation["metric"]: additional_prompts[args.add_prompt][0]
            .replace("{}", label_lists[annotation["metric"]]["labels_only"])
            .replace(
                "{therefore}", label_lists[annotation["metric"]]["therefore"]
            )
            .replace(
                "{correct}",
                label_lists[annotation["metric"]]["correct"],
            )
            + annotation["prompt"]
            + additional_prompts[args.add_prompt][1]
            .replace("{}", label_lists[annotation["metric"]]["labels_only"])
            .replace(
                "{therefore}", label_lists[annotation["metric"]]["therefore"]
            )
            .replace(
                "{correct}",
                label_lists[annotation["metric"]]["correct"],
            )
            for annotation in data["annotations"]
            if annotation["prompt"]
        }

        if not prompts:
            continue
        # slot in instances from dataset
        dataset = {
            metric: [
                replace_instance(metric_prompt, instance)
                for instance in data["instances"]
            ]
            for metric, metric_prompt in prompts.items()
        }
        # collect responses
        responses = {
            metric: model.generate_responses(
                metric_prompt,
                batch_size,
                system_prompt=system_prompt.replace(
                    "{}", label_lists[metric]["labels_only"]
                )
                .replace(
                    "{therefore}",
                    label_lists[metric]["therefore"],
                )
                .replace(
                    "{correct}",
                    label_lists[metric]["correct"],
                ),
            )
            if system_prompt
            else model.generate_responses(
                metric_prompt, batch_size, system_prompt=system_prompt
            )
            for metric, metric_prompt in dataset.items()
        }
        # store responses in file
        for i in range(len(data["instances"])):
            for metric in dataset:
                data["instances"][i]["annotations"][metric][
                    args.model
                ] = responses[metric][i]
        # write json with responses to file
        model_name_without_org = args.model.split("/")[-1]
        current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        with open(
            f"{args.results_dir}/{dataset_name}_{model_name_without_org}-sp{args.system_prompt}-ap{args.add_prompt}_{current_time}.json",
            "w",
            encoding="utf-8",
        ) as outfile:
            json.dump(data, outfile)
