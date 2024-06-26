import json
from statistics import mean 
import pandas as pd
from copy import deepcopy


PROMPT = "Your task is to evaluate the quality of machine translation output at the segment level, "\
    "where a segment may consist of one or more sentences. You will assess the overall quality of each "\
    "translation segment and assign a rating on a scale from 0 to 6.\n\nRating Scale:\n\n"\
    "0: Nonsense/No meaning preserved: Nearly all information is lost between the translation and source. Grammar is irrelevant.\n"\
    "2: Some Meaning Preserved: The translation preserves some of the meaning of the source but misses significant parts. The narrative is hard to follow due to fundamental errors. Grammar may be poor.\n"\
    "4: Most Meaning Preserved and Few Grammar Mistakes: The translation retains most of the meaning of the source. It may have some grammar mistakes or minor contextual inconsistencies.\n"\
    "6: Perfect Meaning and Grammar: The meaning of the translation is completely consistent with the source and the surrounding context (if applicable). The grammar is also correct.\n"\
    "Intermediate levels 1, 3 and 5 can also be chosen as ratings.\n"\
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


SCHEMA = {
            "dataset": "WMT-Human-Evaluation (Freitag et al., Transactions of the Association for Computational Linguistics 2021)",
            "dataset_url": "https://github.com/google/wmt-mqm-human-evaluation",
            "expert_annotator": "true",
            "original_prompt": False,
            "annotations": [
                { 
                    "metric": "quality",
                    "category" : "graded",
                    "prompt": PROMPT,
                    "worst": 0,
                    "best": 6
                }
            ],
            "instances": []
        }


INSTANCE_SCHEMA = { 
            "id": int,
            "instance": {
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


def convert_data(data_file_list,
                output_file):
    """
    Convert .tsv file with annotations from WMT20 datasets into .json schema.

                Parameters:
                        data_file_list (list(str)): path to .tsv WMT20 annotation files
                        output_file (str): path to save converted .json
    """

    datasets = []

    instances_count = 0

    for f in data_file_list:

        schema = deepcopy(SCHEMA)

        raw_data = pd.read_csv(f, sep='\t')

        number_of_segments = raw_data['seg_id'].max()

        for i in range(1, number_of_segments+1):
            filtered_seg = raw_data[raw_data['seg_id'] == i]

            # Choose as reference translation the human translation that obtained the 
            # best average rating by professional translators
            human_translations = filtered_seg[filtered_seg['system'].str.contains('Human')]
            
            # Exclude NaNs
            human_translations = human_translations.dropna()
            if len(human_translations) > 0:
                reference_translation = human_translations.groupby('target')['score'].mean().idxmax()

            for sys in set(filtered_seg[~filtered_seg['system'].str.contains('Human')]['system']):
                filtered = filtered_seg[filtered_seg['system'] == sys]
                annotation_dict = deepcopy(INSTANCE_SCHEMA)
                source = filtered["source"].to_list()[0]
                target = filtered["target"].to_list()[0]
                # Exclude NaNs
                if pd.isna(source) or pd.isna(target):
                    continue
                else:
                    target = target.replace("<v>", "").replace("</v>", "")
                    human_scores = filtered["score"].to_list()
                    annotation_dict["id"] = instances_count
                    annotation_dict["instance"]["source"] = source
                    annotation_dict["instance"]["reference"] = reference_translation
                    annotation_dict["instance"]["translation"] = target
                    annotation_dict["annotations"]["quality"]["mean_human"] = mean(human_scores)
                    annotation_dict["annotations"]["quality"]["individual_human_scores"] = human_scores
                    schema["instances"].append(annotation_dict)
                    instances_count += 1
            
        datasets.append(schema)
                
    print(f"Number of instances in converted datasets: {len(datasets[0]["instances"])}, {len(datasets[1]["instances"])}")
    
    with open(f"{output_file}_en_de.json", 'w', encoding='utf-8') as o:
        json.dump(datasets[0], o, ensure_ascii=False, indent=4)
    with open(f"{output_file}_zh_en.json", 'w', encoding='utf-8') as o:
        json.dump(datasets[1], o, ensure_ascii=False, indent=4)


if __name__=='__main__':

    
    convert_data(data_file_list=["original_data/psqm_newstest2020_ende.tsv", "original_data/psqm_newstest2020_zhen.tsv"],
                            output_file="wmt-human")
