import json
import tiktoken


dataset_names = {
    # prompt needs fixing
    # "chatbot_arena_conversations": "data",
    "cola": "cola",
    "cola-grammar": "cola-grammar",
    "dailydialog-acceptability": "data",
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
    "translated_bnc-2": "translated_bnc_MOP2",
    "translated_bnc-4": "translated_bnc_MOP4",
    "translated_bnc-10": "translated_bnc_MOP100",
    "wmt-06-de-en": "wmt-06-de-en",
    "wmt-06-en-de": "wmt-06-en-de",
    "wmt-06-fr-en": "wmt-06-fr-en",
    "wmt-06-en-fr": "wmt-06-en-fr",
    "wmt-06-es-en": "wmt-06-es-en",
    "wmt-06-en-es": "wmt-06-en-es",
    #"wmt-human": "wmt-human",
}

double_names = {
    "roscoe-cosmos": "roscoe",
    "roscoe-drop": "roscoe",
    "roscoe-esnli": "roscoe",
    "roscoe-gsm8k": "roscoe",
    "toxic_chat-train": "toxic_chat",
    "toxic_chat-test": "toxic_chat",
    "translated_bnc-2": "translated_bnc",
    "translated_bnc-4": "translated_bnc",
    "translated_bnc-10": "translated_bnc",
    "wmt-06-de-en": "wmt-06",
    "wmt-06-en-de": "wmt-06",
    "wmt-06-fr-en": "wmt-06",
    "wmt-06-es-en": "wmt-06",
    "wmt-06-en-es": "wmt-06",
    "wmt-06-en-fr": "wmt-06",
}

lower_batch_size = [
    "roscoe-cosmos",
    "roscoe-drop",
    "roscoe-esnli",
    "roscoe-gsm8k",
    "summeval",
    "topical_chat",
    "wmt-human",
]

lowest_batch_size = ["newsroom"]

api_models = [
    "gpt-4o",
    "gpt-3.5-turbo-0125",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.0-pro-latest",
    "gemini-1.0-pro"
]

additional_prompts = {
    0: " Do not output anything else. Do not explain or justify your answer. Please output only one number, and nothing else. Answer with one of {}.",
    1: " Answer with one of {}. Do not explain your answer."
}

system_prompts = {
    0: "You are a helpful but laconic assistant, skilled in answering language questions very briefly.",
}

def replace_instance(prompt, instance):
    if type(instance["instance"]) == str:
        return prompt.replace("{{ instance }}", instance["instance"])
    elif type(instance["instance"]) == dict:
        for part in instance["instance"]:
            prompt = prompt.replace(
                f"{{ {part} }}", instance["instance"][part]
            )
        return prompt


add_prompt = 1

total_examples = 0
total_tokens = 0
# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")

for dataset_name, filename in dataset_names.items():

    resolved_name = double_names[dataset_name] if dataset_name in double_names else dataset_name
    filepath = f"data/{resolved_name}/{filename}.json"

    # load dataset
    with open(filepath, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    def get_label_list(annotation):
        """
        Given an annotation, returns the list of corresponding labels
        """
        if 'labels_list' in annotation:
            return ', '.join(annotation['labels_list'])
        else:
            lowest_label = min(annotation["best"], annotation["worst"])
            highest_label = max(annotation["best"], annotation["worst"])
            return ', '.join([str(i) for i in range(lowest_label, highest_label)])
        
    prompts = {
        annotation["metric"]: annotation["prompt"] + additional_prompts[add_prompt].format(get_label_list(annotation))
        for annotation in data["annotations"]
        if annotation["prompt"]
    }

    if not prompts:
        continue

    dataset = {metric: [replace_instance(metric_prompt, instance) for instance in data["instances"]]
        for metric, metric_prompt in prompts.items()}

    dataset_tokens = 0    
    dataset_examples = 0
    print(dataset_name)
    for metric, metric_prompts in dataset.items():
        metric_tokens = 0
        metric_examples = 0
        for prompt in metric_prompts:
            metric_examples += 1

            # # adapted from https://platform.openai.com/docs/guides/text-generation/managing-tokens
            # num_tokens = 0
            # num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            # num_tokens += len(enc.encode(prompt))
            # num_tokens += 2  # every reply is primed with <im_start>assistant

            metric_tokens += len(enc.encode(prompt)) + 7

        print(f'{metric_examples}\t{metric_tokens:,}\t{metric}')
        dataset_tokens += metric_tokens 
        dataset_examples += metric_examples
    
    print(f'{dataset_examples}\t{dataset_tokens:,}\t{dataset_name} TOTAL')
    print()
    total_examples += dataset_examples 
    total_tokens += dataset_tokens
print(f'{total_examples}\t{total_tokens:,}\tALL DATASETS TOTAL')
print(f"Estimated cost: {total_examples /1000000 * 15} + {total_tokens / 1000000 * 5} = {total_examples /1000000 * 15 + total_tokens / 1000000 * 5}")
