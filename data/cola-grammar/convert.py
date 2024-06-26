import json
import pandas as pd
from copy import deepcopy
from prompts import *


def convert_data(data_file,
                output_file):
    """
    Convert .tsv file with annotations from CoLA dataset into .json schema.

                Parameters:
                        data_file (str): path to .tsv CoLA annotation file
                        output_file (str): path to save converted .json
    """

    # Get metric names from file
    metrics = list(pd.read_csv(data_file, sep='\t').columns)[4:]

    annotations = [
        { 
            "metric": metric,
            "category" : "categorical",
            "prompt": PROMPTS[idx],
            "labels_list" : [
                "Yes",
                "No"
            ]
        } for idx, metric in enumerate(metrics)
    ]
    

    schema = {
            "dataset": "Grammatically Annotated CoLA (Warstadt and Bowman, arXiv 2019)",
            "dataset_url": "https://nyu-mll.github.io/CoLA/#grammatical_annotations",
            "expert_annotator": "true",
            "original_prompt": False,
            "annotations": annotations,
            "instances": []
        }


    instance_schema = {
        "id": int,
        "instance": "",
        "annotations": {} 
    }

    for metric in metrics:
        instance_schema["annotations"][metric] = {
            "majority_human": "",
            "individual_human_scores": []
        }
    
    instances_count = 0

    raw_data = pd.read_csv(data_file, sep='\t')

    for index, row in raw_data.iterrows():
        instance = row["Sentence"]
        annotation_dict = deepcopy(instance_schema)
        annotation_dict["id"] = instances_count
        annotation_dict["instance"] = instance
        for metric in metrics:
            annotation = row[metric]
            annotation = "Yes" if annotation == 1 else "No"
            annotation_dict["annotations"][metric]["majority_human"] = annotation
            annotation_dict["annotations"][metric]["individual_human_scores"].append(annotation)
        schema["instances"].append(annotation_dict)
        instances_count += 1

    print(f"Number of instances in converted dataset: {len(schema["instances"])}")

    with open(output_file, 'w', encoding='utf-8') as o:
        json.dump(schema, o, indent=4)

if __name__=='__main__':

    
    convert_data(data_file="original_data/CoLA_grammatical_annotations_minor_features.tsv",
                            output_file="cola-grammar.json")
    
    
