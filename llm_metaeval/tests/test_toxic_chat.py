import json
import os

import pytest

from llm_metaeval.data.dataclasses import Dataset

# FIXTURES_FOLDER = os.path.join(os.getcwd(), "llm_metaeval", "tests", "fixtures")


@pytest.fixture
def raw_dataset():
    with open(os.path.join(os.getcwd(), "data", "toxic_chat", "toxic_chat_train.json"), "r") as f:
        data = json.load(f)

    return data


def test_parse_python_object(raw_dataset):

    dataset = Dataset.model_validate(raw_dataset)


    assert len(dataset.instances) > 2000


    assert len(dataset.annotations) == 2


