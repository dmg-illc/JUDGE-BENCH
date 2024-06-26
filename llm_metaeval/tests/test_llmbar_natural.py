import json
import os
import pytest
from llm_metaeval.data.dataclasses import Dataset


@pytest.fixture
def raw_dataset():
    with open(os.path.join(os.getcwd(), "data", "llmbar", "data-natural.json"), "r") as f:
        data = json.load(f)
    return data


def test_parse_python_object(raw_dataset):

    dataset = Dataset.model_validate(raw_dataset)

    # 100 instances available in Natural, 319 in adversarial
    assert len(dataset.instances) == 100

    # only one annotation available: "quality_single_turn"
    assert len(dataset.annotations) == 1

    # check instance ids
    assert dataset.instances[0].id == "Natural_0"
