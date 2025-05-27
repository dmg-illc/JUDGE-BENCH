import argparse
import json
import glob
import os
import re
import time
from random import randrange
import random

import krippendorff
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

date = time.strftime("%d-%m-%Y")
uneven_human_judgements = False
tasks_dataset_names = {
    "cola": "Acceptability",
    "cola-grammar": "Acceptability",
    "dailydialog-acceptability": "Acceptability",
    "newsroom": "Summarisation",
    "persona-chat": "Dialogue",
    "persona_chat": "Dialogue",
    "qags": "Summarisation",
    "roscoe-cosmos": "Reasoning",
    "roscoe-drop": "Reasoning",
    "roscoe-esnli": "Reasoning",
    "roscoe-gsm8k": "Reasoning",
    "summeval": "Summarisation",
    "switchboard-acceptability": "Acceptability",
    "topical_chat": "Dialogue",
    "topical-chat": "Dialogue",
    "wmt-human_en_de": "Translation",
    "wmt-human_zh_en": "Translation",
    "wmt-human-en-de": "Translation",
    "wmt-human-zh-en": "Translation",
    "wmt-23_en_de": "Translation",
    "wmt-23_zh_en": "Translation",
    "wmt-23-en-de": "Translation",
    "wmt-23-zh-en": "Translation",
    "inferential-strategies": "Reasoning",
    "medical-safety": "Toxicity \ Safety",
    "toxic_chat": "Toxicity \ Safety",
    "toxic-chat": "Toxicity \ Safety",
    "dices-990": "Toxicity \ Safety",
    "dices_990": "Toxicity \ Safety",
    "dices_350_expert": "Toxicity \ Safety",
    "dices-350-expert": "Toxicity \ Safety",
    "dices_350_crowdsourced": "Toxicity \ Safety",
    "dices-350-crowdsourced": "Toxicity \ Safety",
    "recipe_crowd_sourcing_data": "Planning",
    "recipe-crowd-sourcing-data": "Planning",
    "llmbar-natural": "Instruction Following",
    "llmbar-adversarial": "Instruction Following",
}
dataset_names = [
    "cola",
    "cola-grammar",
    "dailydialog-acceptability",
    "dices_990",
    "dices_350_expert",
    "dices_350_crowdsourced",
    "newsroom",
    "persona_chat",
    "qags",
    "roscoe-cosmos",
    "roscoe-drop",
    "roscoe-esnli",
    "roscoe-gsm8k",
    "summeval",
    "switchboard-acceptability",
    "topical_chat",
    "toxic_chat",
    "wmt-human_en_de",
    "wmt-human_zh_en",
    "wmt-23_en_de",
    "wmt-23_zh_en",
    "inferential-strategies",
    "medical-safety",
    "recipe_crowd_sourcing_data",
    "llmbar-natural",
    "llmbar-adversarial",
    # "chatbot_arena_conversations",
]

model_names = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "berkeley-nest/Starling-LM-7B-alpha",
    "CohereForAI/c4ai-command-r-v01",
    "CohereForAI/c4ai-command-r-plus",
    "allenai/OLMo-7B-0724-Instruct-hf",
    "gpt-3.5-turbo-0125",
    "claude-3-haiku-20240307",
    "gemini-1.5-flash-latest",
    "gpt-4o",
]


def get_files_with_responses(results_dir, dataset, model=None):
    prefix = f"{results_dir}/{dataset}_" + (
        "*" if model is None else f'{model.split("/")[-1]}-spNone-ap*'
    )
    files = sorted(glob.glob(prefix), key=os.path.getctime)
    # switch between files with regular and cot prompts
    files = [
        file
        for file in files
        if "regular" in file
        # if cot" in file
    ]
    # remove results from "claude haiku"
    files = [file for file in files if "haiku" not in file]
    return files if model is None else [files[-1]]


def save_results(results, path):
    with open(path, "w") as fh:
        json.dump(results, fh)
    fh.close()
    print(f"...complete. saved results to: {path}")


def evaluate(set_h, set_m, set_all_h, valid_counter, type, expert, task):
    # 1. correlation between human and model responses
    valid_set_h, valid_set_m = [], []
    for idx in range(len(set_h)):
        if set_m[idx] is not None:
            valid_set_h.append(set_h[idx])
            valid_set_m.append(set_m[idx])
    try:
        correlation_p = pearsonr(valid_set_h, valid_set_m)
        correlation_s = spearmanr(valid_set_h, valid_set_m)
        correlation_k = kendalltau(valid_set_h, valid_set_m)
    except ValueError:  # no valid responses from the model
        correlation_p, correlation_s, correlation_k = [[float("nan")] * 2] * 3
    try:
        kappa_score = cohen_kappa_score(valid_set_h, valid_set_m)
        if valid_set_h == valid_set_m:
            kappa_score = 1
    except ValueError:  # no kappa for numerical data
        kappa_score = float("nan")

    # 2. agreement between responses of humans
    all_equal = True
    for i in range(1, len(set_all_h)):
        if set_all_h[i] != set_all_h[i - 1]:
            all_equal = False
            break
    if len(set_all_h) < 2:  # no agreement if only one rating per instance
        agreement = np.nan
    elif all_equal:
        agreement = 1
    elif type == "categorical":
        agreement = krippendorff.alpha(
            reliability_data=set_all_h, level_of_measurement="nominal"
        )
    elif type == "graded":
        agreement = krippendorff.alpha(
            reliability_data=set_all_h, level_of_measurement="ordinal"
        )
    elif type == "continuous":
        agreement = krippendorff.alpha(
            reliability_data=set_all_h, level_of_measurement="interval"
        )

    return {
        "corr_coeff": {
            "pearson": correlation_p[0],
            "spearman": correlation_s[0],
            "kendall": correlation_k[0],
        },
        "p_value": {
            "pearson": correlation_p[1],
            "spearman": correlation_s[1],
            "kendall": correlation_k[1],
        },
        "kappa_score": kappa_score,
        "total_responses": len(set_h),
        "valid_responses": valid_counter,
        "krippendorff_alpha": agreement,
        "type": type,
        "expert": expert,
        "task": task,
    }


def extract_answer(
    response, category, labels_list, CoT_prompting=False, dataset=None
):
    response = response.strip().lower()

    if CoT_prompting:
        answer = None
        response = response.replace("**", "")
        if category == "categorical" and answer not in labels_list:
            for label in labels_list:
                if f"therefore, {label} is correct." in response:
                    answer = label
                    break
        elif category == "graded" or category == "continuous":
            for label in range(int(labels_list[0]), int(labels_list[1]) + 1):
                if dataset.startswith("recipe") and response.endswith(
                    f"therefore, {label} is correct."
                ):
                    answer = label
                    break
                elif f"therefore, {label} is correct." in response:
                    answer = label
                    break

        if answer == None:
            print(
                f"[INVALID RESPONSE begins]: {response} [INVALID RESPONSE ends]\n"
            )
        return answer, "valid" if answer != None else "non-valid"

    if category == "graded":
        search_for = rf"(?<!\d)[{labels_list[0]}-{labels_list[1]}](?!\d)"
        match_found = re.search(search_for, response)
        if match_found is not None:
            return float(match_found[0]), "valid"
        else:
            # replace with a dummy response:
            return (
                float(randrange(labels_list[0], labels_list[1])),
                "non-valid",
            )
    elif category == "continuous":
        search_for = rf"[-+]?[0-9]*\.?[0-9]+"
        match_found = re.search(search_for, response)
        if match_found is not None:
            return float(match_found[0]), "valid"
        else:
            # replace with a dummy response:
            return (
                float(randrange(int(labels_list[0]), int(labels_list[1]))),
                "non-valid",
            )
    elif category == "categorical":
        for label in labels_list:
            search_for = rf"\b{label.strip().lower()}\b"
            match_found = re.search(search_for, response)
            if match_found is not None:
                return match_found[0], "valid"
        return random.choice(labels_list), "non-valid"
    return None


def get_responses(file):
    global uneven_human_judgements
    responses = {}
    with open(file, "r") as fh:
        responses = json.load(fh)
    fh.close()
    try:
        expert = responses["expert_annotator"]
    except:
        expert = "uknown"
    assert (
        len(responses["instances"]) > 0
    ), f"[ERROR] no instances found in file: {file}"

    run_details = responses["run_details"]
    model_name_with_org = run_details["model"]
    dataset = responses["dataset"].split(" ")[0]

    model_name_for_results = model_name_with_org.split("/")[-1]
    ap_id = (
        None
        if "additional_prompt_id" not in run_details
        else run_details["additional_prompt_id"]
    )
    if ap_id:
        model_name_for_results += f" (AP: {ap_id})"
    print(f"\t{model_name_for_results}")

    valid_categories = ["graded", "categorical", "continuous"]
    processed_responses = {}
    for annotation in responses["annotations"]:
        assert (
            "category" in annotation
        ), "[ERROR] failed to determine data type"
        metric, category = annotation["metric"], annotation["category"]
        assert (
            category in valid_categories
        ), f"[ERROR] cannot process {category} category responses yet"
        labels_list = (
            [annotation["worst"], annotation["best"]]
            if category in ["graded", "continuous"]
            else list(map(str.lower, annotation["labels_list"]))
        )
        n_humans = max(
            [
                len(instance["annotations"][metric]["individual_human_scores"])
                for instance in responses["instances"]
            ]
        )
        human_responses, model_responses, all_human_responses = (
            [],
            [],
            [[] for _ in range(n_humans)],
        )
        num_valid_responses = 0
        for instance in responses["instances"]:
            if model_name_with_org not in instance["annotations"][metric]:
                continue
            judgement_type = (
                "mean_human"
                if category in ["graded", "continuous"]
                else "majority_human"
            )
            human_response = instance["annotations"][metric][judgement_type]
            model_response, validity = extract_answer(
                instance["annotations"][metric][model_name_with_org],
                category,
                labels_list,
                CoT_prompting=("cot" in ap_id),
                dataset=dataset,
            )
            if validity == "valid":
                num_valid_responses += 1
            if (
                not uneven_human_judgements
                and len(
                    instance["annotations"][metric]["individual_human_scores"]
                )
                != n_humans
            ):
                uneven_human_judgements = True
            if instance["annotations"][metric]["individual_human_scores"]:
                for h_id, h_response in enumerate(
                    instance["annotations"][metric]["individual_human_scores"]
                ):
                    all_human_responses[h_id].append(h_response)
                for other_h_id in range(h_id + 1, n_humans):
                    all_human_responses[other_h_id].append(np.nan)

            if category == "categorical":
                human_response = labels_list.index(human_response.lower())
                model_response = (
                    labels_list.index(model_response)
                    if model_response is not None
                    else model_response
                )
                for h_id in range(len(all_human_responses)):
                    if type(all_human_responses[h_id][-1]) == str:
                        all_human_responses[h_id][-1] = labels_list.index(
                            all_human_responses[h_id][-1].lower()
                        )
            human_responses.append(human_response)
            model_responses.append(model_response)

        assert len(human_responses) == len(
            model_responses
        ), f"[ERROR] {len(human_responses)} human responses and {len(model_responses)} model responses"
        processed_responses[metric] = (
            human_responses,
            model_responses,
            all_human_responses,
            num_valid_responses,
            category,
            expert,
        )

    return processed_responses, model_name_for_results


def process_files(dataset, files, results_dir):
    # step 1: process and parse raw responses
    responses_for_dataset = {}
    for file in files:
        processed_responses, model = get_responses(file)
        responses_for_dataset[f'{dataset.replace("_", "-")} | {model}'] = (
            processed_responses
        )

    # step 2: evaluate using different metrics
    results = {}
    for key, responses in responses_for_dataset.items():
        dataset, model = key.split(" | ")
        task = tasks_dataset_names[dataset]
        results[model] = {}
        for metric, (
            set_h,
            set_m,
            set_all_h,
            valid_counter,
            category,
            expert,
        ) in responses.items():
            results[model][metric] = evaluate(
                set_h,
                set_m,
                set_all_h,
                valid_counter,
                type=category,
                expert=expert,
                task=task,
            )

    # step 3: save to disk
    path = f"{results_dir}/eval/{date}"
    os.makedirs(path, exist_ok=True)
    save_as = (
        f"{path}/{dataset}.json"
        if len(files) > 1
        else f"{path}/{dataset}_{model}.json"
    )
    save_results(results, save_as)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="utility to evaluate responses generated by LLMs for datasets containing human judgements:",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-rf",
        "--responses_file",
        type=str,
        default=None,
        help="path to the model responses file (in JSON format)",
        metavar="",
    )
    parser.add_argument(
        "-rd",
        "--results_dir",
        type=str,
        default="results",
        help="path to the results directory",
        metavar="",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="ALL",
        choices=dataset_names,
        help=f'select a dataset from:\n[{", ".join(dataset_names)}]',
        metavar="",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        choices=model_names,
        help=f'select a model from:\n[{", ".join(model_names)}]',
        metavar="",
    )
    args = parser.parse_args()

    datasets_to_eval = (
        dataset_names if args.dataset == "ALL" else [args.dataset]
    )
    for dataset in datasets_to_eval:
        print(f"\nevaluating responses for [{dataset}] dataset from models:")
        uneven_human_judgements = False
        # step 0: obtain paths to files with model responses
        files_to_process = []
        try:
            if args.responses_file is not None:
                files_to_process.append(args.responses_file)
            else:
                files_to_process = get_files_with_responses(
                    results_dir=args.results_dir,
                    dataset=dataset,
                    model=args.model,
                )
        except Exception as e:
            print(f"[ERROR] unable to locate files with model responses: {e}")
            raise

        assert len(files_to_process) > 0, "[ERROR] no files to process!"
        process_files(dataset, files_to_process, args.results_dir)
        if uneven_human_judgements:
            print(
                f"[ALERT]: uneven number of human responses in [{dataset}]!\n"
            )
