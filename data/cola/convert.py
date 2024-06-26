import json
import pandas as pd
from copy import deepcopy

def convert_data(data_file_list,
                output_file):
    """
    Convert .tsv file with annotations from CoLA dataset into .json schema.

                Parameters:
                        data_file_list (list(str)): path to .tsv CoLA annotation files
                        output_file (str): path to save converted .json
    """
    
    schema = {
            "dataset": "The Corpus of Linguistic Acceptability (Warstadt et al., Transactions of the Association for Computational Linguistics 2019)",
            "dataset_url": "https://nyu-mll.github.io/CoLA/",
            "expert_annotator": "true",
            "original_prompt": False,
            "annotations": [
                { "metric": "grammaticality",
                "category" : "categorical",
                "prompt": "Given the following sentence, determine if it is grammatically correct or not. Write 'Yes' if it is grammatical, and 'No' if it is not:\n\n{{ instance }}",
                "labels_list" : [
                    "Yes",
                    "No"
                ]
                }
            ],
            "instances": []
        }

    instance_schema = { 
            "id": int,
            "instance": "",
            "annotations": {
                "grammaticality": {
                    "majority_human": "",
                    "individual_human_scores": []
                }
            }
        }
    
    instances_count = 0

    for f in data_file_list:

        raw_data = pd.read_csv(f, header=None, sep='\t')

        for index, row in raw_data.iterrows():
            instance = row.iloc[3]
            annotation = row.iloc[1]
            annotation = "Yes" if annotation == 1 else "No"
            annotation_dict = deepcopy(instance_schema)
            annotation_dict["id"] = instances_count
            annotation_dict["instance"] = instance
            annotation_dict["annotations"]["grammaticality"]["majority_human"] = annotation
            annotation_dict["annotations"]["grammaticality"]["individual_human_scores"].append(annotation)
            schema["instances"].append(annotation_dict)
            instances_count += 1

    print(f"Number of instances in converted dataset: {len(schema["instances"])}")

    with open(output_file, 'w', encoding='utf-8') as o:
        json.dump(schema, o, indent=4)

if __name__=='__main__':

    # Training data with annotations, should we include this?
    #convert_data(data_tsv="./original_data/in_domain_train.tsv",
    #                        output_file="./in_domain_train.json")
    
    convert_data(data_file_list=["original_data/in_domain_dev.tsv", "original_data/out_of_domain_dev.tsv"],
                            output_file="cola.json")
    
    
