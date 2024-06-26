import json
import os
import pytest
from llm_metaeval.data.dataclasses import Dataset


@pytest.fixture
def raw_dataset():
    with open(os.path.join(os.getcwd(), "data", "switchboard-acceptability", "data.json"), "r") as f:
        data = json.load(f)

    return data


def test_parse_python_object(raw_dataset):

    dataset = Dataset.model_validate(raw_dataset)

    # 100 instances available
    assert len(dataset.instances) == 100

    # only one annotations available: "acceptability"
    assert len(dataset.annotations) == 1

    assert dataset.instances[0].id == 1

    assert dataset.instances[22].annotations['acceptability'].majority_human == 2

    assert dataset.instances[69].annotations['acceptability'].individual_human_scores == [1, 1, 1, 1]

    assert dataset.instances[12].annotations['acceptability'].mean_human == 1.5
