# -*- coding: utf-8 -*-
"""

Convert Dailydialog dialogue acceptability judgements (Wallbridge et al., Interspeech 2022).

Original file is located at
    https://data.cstr.ed.ac.uk/sarenne/INTERSPEECH2022/switchboard_results_is.csv
"""

import random
import numpy as np
import pandas as pd
from collections import Counter

from llm_metaeval.data.dataclasses import (
    GradedAnnotation,
    GradedAnnotationScores,
    Dataset,
    Instance,
)


DATA_URL = "https://data.cstr.ed.ac.uk/sarenne/INTERSPEECH2022/dailydailog_results_is.csv"


if __name__ == "__main__":
    data = pd.read_csv(DATA_URL)
    data.rename(columns={data.columns[0]: "id" }, inplace = True)

    def majority_vote(scores):
        random.shuffle(scores)
        return Counter(scores).most_common(1)[0][0]
    
    random.seed(0)
    speaker_ids = ["A", "B"]
    instances = []
    for _, row in data.iterrows():
        instance = ""
        scores = eval(row.all_score)
        context_text = eval(row.context_text)
        for i, turn in enumerate(context_text):
            instance += f"{speaker_ids[i % 2]}: {turn} "
        instance += f"{speaker_ids[len(context_text) % 2]}: {row.response_text}"

        instances.append({
            "id": int(row.id) + 1,
            "text": instance,
            "annotations": {
                "acceptability": {
                    "mean_human": np.mean(scores),
                    "individual_human_scores": scores,
                    "majority_human": majority_vote(scores)
                }
            }
        })

    dataset = Dataset(
        dataset="Dailydialog dialogue acceptability judgements (Wallbridge et al., Interspeeech 2022)",
        dataset_url="https://data.cstr.ed.ac.uk/sarenne/INTERSPEECH2022/dailydailog_results_is.csv",
        expert_annotator="false",
        original_prompt=False,
        annotations=[
            GradedAnnotation(
                metric="acceptability",
                category="graded",
                prompt="On a scale of 1 (very unlikely) to 5 (very likely), how plausible is it that the last response belongs to the dialogue? {{ instance }}",
                worst=1,
                best=5
            )
        ],
        instances=[
            Instance(
                id=instance["id"],
                instance=instance["text"],
                annotations={
                    "acceptability": GradedAnnotationScores(
                        mean_human=instance["annotations"]["acceptability"]["mean_human"],
                        individual_human_scores=instance["annotations"]["acceptability"]["individual_human_scores"],
                        majority_human=instance["annotations"]["acceptability"]["majority_human"]
                    )
                }
            ) for instance in instances
        ]
    )

    with open("data.json", "w") as file:
        file.write(dataset.model_dump_json(indent=4))
