import json
import os

import pytest

from llm_metaeval.data.dataclasses import Dataset

FIXTURES_FOLDER = os.path.join(os.getcwd(), "llm_metaeval", "tests", "fixtures")


@pytest.fixture
def raw_dataset():
    with open(f"{FIXTURES_FOLDER}/example.json", "r") as f:
        data = json.load(f)

    return data


def test_parse_python_object(raw_dataset):

    dataset = Dataset.model_validate(raw_dataset)

    # only two instances available
    assert len(dataset.instances) == 2

    # only two annotations available: "grammaticality" and "fluency"
    assert len(dataset.annotations) == 2

    assert dataset.instances[0].id == 1
