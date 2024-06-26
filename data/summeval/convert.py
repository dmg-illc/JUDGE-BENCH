import json
from copy import deepcopy

# Prompts for each type of annotation
COHERENCE = "You will be given one summary written for a news article.\n\nYour task is to rate the summary on one metric.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nEvaluation Criteria:\n\nCoherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby \"the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.\"\n\nEvaluation Steps:\n\n1. Read the news article carefully and identify the main topic and key points.\n2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.{{ instance }}\n\n\nEvaluation Form (scores ONLY):\n\n- Coherence:"
CONSISTENCY= "You will be given a news article. You will then be given one summary written for this article.\n\nYour task is to rate the summary on one metric.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\n\nEvaluation Criteria:\n\nConsistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. \n\nEvaluation Steps:\n\n1. Read the news article carefully and identify the main facts and details it presents.\n2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.\n3. Assign a score for consistency based on the Evaluation Criteria.{{ instance }}\n\n\nEvaluation Form (scores ONLY):\n\n- Consistency:"
FLUENCY = "You will be given one summary written for a news article.\n\nYour task is to rate the summary on one metric.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\n\nEvaluation Criteria:\n\nFluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.\n\n- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.\n- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.\n- 3: Good. The summary has few or no errors and is easy to read and follow.{{ instance }}\n\n\nEvaluation Form (scores ONLY):\n\n- Fluency:"
RELEVANCE = "You will be given one summary written for a news article.\n\nYour task is to rate the summary on one metric.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nEvaluation Criteria:\n\nRelevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.\n\nEvaluation Steps:\n\n1. Read the summary and the source document carefully.\n2. Compare the summary to the source document and identify the main points of the article.\n3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.\n4. Assign a relevance score from 1 to 5.{{ instance }}\n\n\nEvaluation Form (scores ONLY):\n\n- Relevance:"


# General schema
SCHEMA = {
    "dataset": "SummEval (Fabbri et al., Transactions of the Association for Computational Linguistics 2021)",
    "dataset_url": "https://github.com/Yale-LILY/SummEval",
    "prompts_url": "https://github.com/nlpyang/geval/tree/main/prompts/summeval",
    "expert_annotator": "unknown",
    "original_prompt": True,
    "annotations": [
        { "metric": "coherence",
        "category": "graded",
        "prompt": COHERENCE,
        "worst": 1,
        "best": 5
        },
        { "metric": "consistency",
        "category": "graded",
        "prompt": CONSISTENCY,
        "worst": 1,
        "best": 5
        },
        { "metric": "fluency",
        "category": "graded",
        "prompt": FLUENCY,
        "worst": 1,
        "best": 3
        },
        { "metric": "relevance",
        "category": "graded",
        "prompt": RELEVANCE,
        "worst": 1,
        "best": 5
        }
    ],
    "instances": []
}


# Schema for a single instance
INSTANCE_SCHEMA = { 
    "id" : int,
    "instance": "",
    "source": "",
    "reference": "",
    "summary": "",
    "annotations": {
        "coherence": {
            "mean_human": "",
            "individual_human_scores": []
        },
        "consistency": {
            "mean_human": "",
            "individual_human_scores": []
        },
        "fluency": {
            "mean_human": "",
            "individual_human_scores": []
        },
        "relevance": {
            "mean_human": "",
            "individual_human_scores": []
        }
    }
}


def apply_instance_template(source, summary):
    """Apply a template to source text and related summary to process them as a single instance
    """
    template = f"\n\n\nExample:\n\n\nSource Text:\n\n{source}\n\nSummary:\n\n{summary}"
    return template


def convert_data(data_file, output_file):
    """
    Convert .json file with annotations from SummEval dataset into .json schema.

            Parameters: 
                    data_file (str): path to .json SummEval annotation file
                    output_file (str): path to save converted .json
    """

    with open(data_file) as f:
        raw_data = json.load(f)
    
    schema = deepcopy(SCHEMA)

    for idx, row in enumerate(raw_data):
        instance = deepcopy(INSTANCE_SCHEMA)
        instance["id"] = idx
        instance["instance"] = apply_instance_template(row["source"], row["system_output"])
        instance["source"] = row["source"]
        instance["reference"] = row["reference"]
        instance["summary"] = row["system_output"]
        instance["annotations"]["coherence"]["mean_human"] = row["scores"]["coherence"]
        instance["annotations"]["coherence"]["individual_human_scores"].append(round(row["scores"]["coherence"]))
        instance["annotations"]["consistency"]["mean_human"] = row["scores"]["consistency"]
        instance["annotations"]["consistency"]["individual_human_scores"].append(round(row["scores"]["consistency"]))
        instance["annotations"]["fluency"]["mean_human"] = row["scores"]["fluency"]
        instance["annotations"]["fluency"]["individual_human_scores"].append(round(row["scores"]["fluency"]))
        instance["annotations"]["relevance"]["mean_human"] = row["scores"]["relevance"]
        instance["annotations"]["relevance"]["individual_human_scores"].append(round(row["scores"]["relevance"]))
        schema["instances"].append(instance)
    
    print(f"Number of instances in converted dataset: {len(schema["instances"])}")

    with open(output_file, 'w', encoding='utf-8') as o:
        json.dump(schema, o, indent=4)

if __name__=='__main__':

    convert_data(data_file="original_data/summeval.json",
                            output_file="summeval.json")

