import json
import os

import pytest

from llm_metaeval.data.dataclasses import Dataset


@pytest.fixture
def raw_dataset():
    
    with open(os.path.join(os.getcwd(), "data", "summeval", "summeval.json"), "r") as f:
        data = json.load(f)

    return data


def test_parse_python_object(raw_dataset):

    dataset = Dataset.model_validate(raw_dataset)

    assert len(dataset.instances) == 1600


