import os
import json
from statistics import mean 
from collections import defaultdict
import pandas as pd
from copy import deepcopy


PROMPT = "Your task is to evaluate the quality of machine translation output at the segment level, "\
    "where a segment may consist of one or more sentences. You will assess the overall quality of each "\
    "translation segment and assign a rating on a scale from 0 to 100.\n\nRating Scale:\n\n"\
    "0: Nonsense/No meaning preserved: Nearly all information is lost between the translation and source. Grammar is irrelevant.\n"\
    "30: Some Meaning Preserved: The translation preserves some of the meaning of the source but misses significant parts. The narrative is hard to follow due to fundamental errors. Grammar may be poor.\n"\
    "60: Most Meaning Preserved and Few Grammar Mistakes: The translation retains most of the meaning of the source. It may have some grammar mistakes or minor contextual inconsistencies.\n"\
    "100: Perfect Meaning and Grammar: The meaning of the translation is completely consistent with the source and the surrounding context (if applicable). The grammar is also correct.\n"\
    "Intermediate levels can also be chosen as ratings.\n"\
    "\n\nEvaluation Criteria:\n\nWhen evaluating the quality of each translation segment, consider the following criteria:\n\n"\
    "Accuracy: How well does the translation convey the original meaning and content of the source text?\n"\
    "Fluency: How natural and idiomatic is the translation in terms of grammar, syntax, and phrasing?\n"\
    "Comprehensibility: How easily can the translation be understood by a native speaker of the target language?\n"\
    "Errors: Are there any errors in grammar, vocabulary, punctuation, or formatting that affect the overall quality of the translation?\n\n"\
    "You will be provided with a source text, a reference human translation of the source text, and a candidate translation that you have to evaluate. "\
    "Use the reference translation to better evaluate the candidate translation.\n\n"\
    "Source Text for Translation:\n{{ source }}\n\n"\
    "Reference Human Translation:\n{{ reference }}\n\n"\
    "Candidate Translation to Evaluate:\n{{ translation }}\n\n"\

PROMPT_PARA_1 = "Assess the translation quality from {{ source_lang }} to {{ target_lang }} relative to the human reference, "\
    "using a rating scale from 0 to 100 that captures translation depth:\n"\
    '- 0: "No aspect of the original message is conveyed in the translation"\n'\
    '- 30: "The translation partially conveys the original message, but important elements are missing or unclear"\n'\
    '- 60: "The translation successfully conveys most of the original content with only slight grammatical errors"\n'\
    '- 100: "The translation is an exact and grammatically perfect reproduction of the original message"\n\n'\
    '{{ source_lang }} source: "{{ source }}"\n'\
    '{{ target_lang }} human reference: "{{ reference }}"\n'\
    '{{ target_lang }} translation: "{{ translation }}"\n'\
    'Translation Score (0-100):'

# Prompt taken from Large Language Models Are State-of-the-Art Evaluators of Translation Quality (Kocmi & Federmann, EAMT 2023)
PROMPT_PARA_2 = "Score the following translation from {{ source_lang }} to {{ target_lang }} **with respect to"\
    "the human reference** on a continuous scale from 0 to 100, where a score of zero means"\
    'no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n'\
    '{{ source_lang }} source: "{{ source }}"\n'\
    '**{{ target_lang }} human reference: "{{ reference }}"**\n'\
    '{{ target_lang }} translation: "{{ target }}"\n'\
    'Score:'

# Prompt taken from Large Language Models Are State-of-the-Art Evaluators of Translation Quality (Kocmi & Federmann, EAMT 2023)
PROMPT_PARA_3 = "Score the following translation from {{ source_lang }} to"\
    "{{ target_lang }} with respect to the human reference on a continuous"\
    'scale from 0 to 100 that starts with "No meaning preserved", goes'\
    'through "Some meaning preserved", then "Most meaning preserved and'\
    'few grammar mistakes", up to "Perfect meaning and grammar".\n\n'\
    '{{ source_lang }} source: "{{ source }}"\n'\
    '**{{ target_lang }} human reference: "{{ reference }}"**\n'\
    '{{ target_lang }} translation: "{{ translation }}"\n'\
    'Score (0-100):'

PROMPT_FSP = "Score the following translation from {{ source_lang }} to {{ target_lang }} **with respect to"\
    "the human reference** on a continuous scale from 0 to 100, where a score of zero means"\
    'no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n'\
    '{{ examples }}'\
    "New example to score:\n\n"\
    'Source: "{{ source }}"\n'\
    'Human reference: "{{ reference }}"\n'\
    'Translation: "{{ translation }}"\n'\
    'Score:'

SCHEMA = {
            "dataset": "WMT-23 (Kocmi et al., Proceedings of the Eighth Conference on Machine Translation 2023)",
            "dataset_url": "https://github.com/google-research/mt-metrics-eval",
            "expert_annotator": "true",
            "original_prompt": False,
            "annotations": [
                { 
                    "metric": "quality",
                    "category": "continuous",
                    "prompt": PROMPT,
                    "prompt_paraphrase_1": PROMPT_PARA_1,
                    "prompt_paraphrase_2": PROMPT_PARA_2,
                    "prompt_paraphrase_3": PROMPT_PARA_3,
                    "prompt_fsp": None,
                    "fsp_examples_id": None,
                    "worst": 0.0,
                    "best": 100.0
                }
            ],
            "instances": []
        }


INSTANCE_SCHEMA = { 
            "id": int,
            "instance": {
                "source_lang": "",
                "target_lang": "",
                "source" : "",
                "reference" : "",
                "translation" : ""
            },
            "annotations": {
                "quality": {
                    "mean_human": "",
                    "individual_human_scores": []
                }
            }
        }


def get_few_shot_examples(data_dict):
    """
    Get few-shot examples for each language pair.

    Parameters:
    data_dict (dict): Dictionary containing dataframes for each language pair.

    Returns:
    few_shot_ids: Dictionary with few-shot example ids for each language pair.
    """
    few_shot_ids = {
        'en-de': [],
        'zh-en': []
    }

    # Process each language pair
    for lang_pair in ['en-de', 'zh-en']:

        # Filter for language pair
        lang_data = data_dict[lang_pair]

        # Sort by score
        sorted_data = lang_data.sort_values('score')

        # Get 2 lowest and 2 highest scoring examples
        low_examples = sorted_data.head(2)
        high_examples = sorted_data.tail(2)

        few_shot_ids[lang_pair].extend(low_examples.index.tolist())
        few_shot_ids[lang_pair].extend(high_examples.index.tolist())

    return few_shot_ids


def update_fsp_prompt(data_df, few_shot_ids):
    """
    Creates few-shot prompt with examples from the provided data.
    
    Args:
        data_df (DataFrame): A DataFrame containing example data for a single language pair.
        few_shot_ids (list): A list of example IDs to include in the prompt.
    
    Returns:
        updated_prompt (str): A string containing the updated prompt with the few-shot examples inserted
    """
    few_shot_text = f"Example scoring:\n\n"
    
    # Add each example to the prompt
    for ex_id in few_shot_ids:
        example = data_df.iloc[ex_id]
        few_shot_text += f"Source: \"{example['source']}\"\n"
        few_shot_text += f"Human Reference: \"{example['reference']}\"\n"
        few_shot_text += f"Translation: \"{example['translation']}\"\n"
        few_shot_text += f"Score: {example['score']}\n\n"
    
    updated_prompt = PROMPT_FSP.replace("{{ examples }}", few_shot_text)
    
    return updated_prompt


def create_df_from_original_data():
    data = {}
    languages = ["en-de", "zh-en"]

    for lang in languages:
        df = defaultdict(list)

        with open(os.path.join("original_data", f"{lang}_source.txt")) as s:
            sources = s.read().split("\n")[:-1]

        with open(os.path.join("original_data", f"{lang}_ref.txt")) as s:
            references = s.read().split("\n")[:-1]

        files = [os.path.join(lang, entry) for entry in os.listdir(os.path.join("original_data", lang))]
        sys_names = [entry.replace(".txt", "") for entry in os.listdir(os.path.join("original_data", lang)) if entry not in ["refA.txt", "synthetic_ref.txt"]]
        translations = {}
        for fil, sys in zip(files, sys_names):
            with open(os.path.join("original_data", fil)) as f:
                translations[sys] = f.read().split("\n")[:-1]

        scores = defaultdict(list)
        with open(os.path.join("original_data", f"{lang}_score.txt")) as f:
            scores_tuple = f.read().split("\n")
        for tupl in scores_tuple[:-1]:
            if tupl.split("\t")[0] != "refA":
                scores[tupl.split("\t")[0]].append(tupl.split("\t")[1])

        for idx, source in enumerate(sources):
            for sys, translation in translations.items():
                if scores[sys][idx] != "None" and references[idx] != "\"":
                    df["system"].append(sys)
                    df["source"].append(source)
                    df["reference"].append(references[idx])
                    df["translation"].append(translation[idx])
                    df["score"].append(float(scores[sys][idx]))

        data[lang] = pd.DataFrame(df)

    return data


def convert_data(data_dict,
                output_file):
    """
    Convert .tsv file with annotations from WMT23 datasets into .json schema.

                Parameters:
                        data_dict (dict): dict with dataframe as keys
                        output_file (str): path to save converted .json
    """
    languages = {
        "en-de": {
            "source_lang" : "English",
            "target_lang" : "German"
        }, 
        "zh-en": {
            "source_lang" : "Chinese",
            "target_lang" : "English"
        }
    }
    few_shot_ids = get_few_shot_examples(data_dict)

    datasets = []

    for dataset_name, df in data_dict.items():        

        schema = deepcopy(SCHEMA)
        updated_fsp_prompt = update_fsp_prompt(df, few_shot_ids[dataset_name])
        schema["annotations"][0]["prompt_fsp"] = updated_fsp_prompt
        schema["annotations"][0]["fsp_examples_id"] = few_shot_ids[dataset_name]

        for index, row in df.iterrows():
            if pd.notna(row).all():  # check if all values in the row are not NaN
                annotation_dict = deepcopy(INSTANCE_SCHEMA)
                annotation_dict["id"] = index
                annotation_dict["instance"]["source"] = row["source"]
                annotation_dict["instance"]["reference"] = row["reference"]
                annotation_dict["instance"]["translation"] = row["translation"]
                annotation_dict["instance"]["source_lang"] = languages[dataset_name]["source_lang"]
                annotation_dict["instance"]["target_lang"] = languages[dataset_name]["target_lang"]
                annotation_dict["annotations"]["quality"]["mean_human"] = row["score"]
                annotation_dict["annotations"]["quality"]["individual_human_scores"].append(row["score"])
                schema["instances"].append(annotation_dict)
            
        datasets.append(schema)
                
    print(f"Number of instances in converted datasets: {len(datasets[0]["instances"])}, {len(datasets[1]["instances"])}")
    
    with open(f"{output_file}_en_de.json", 'w', encoding='utf-8') as o:
        json.dump(datasets[0], o, ensure_ascii=False, indent=4)
    with open(f"{output_file}_zh_en.json", 'w', encoding='utf-8') as o:
        json.dump(datasets[1], o, ensure_ascii=False, indent=4)


if __name__=='__main__':

    data_dict = create_df_from_original_data()
    
    convert_data(data_dict=data_dict, output_file="wmt-23")
