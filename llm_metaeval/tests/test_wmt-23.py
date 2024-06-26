import json
import os

import pytest

from llm_metaeval.data.dataclasses import Dataset


@pytest.fixture(params=["wmt-23_en_de.json", "wmt-23_zh_en.json"])
def raw_dataset(request):
    
    dataset_name = request.param.split(".")[0]
    
    with open(os.path.join(os.getcwd(), "data", "wmt-23", request.param), "r") as f:
        data = json.load(f)
    
    return data, dataset_name


def test_parse_python_object(raw_dataset):

    data, dataset_name = raw_dataset
    
    dataset = Dataset.model_validate(data)

    if "en_de" in dataset_name:
        assert len(dataset.instances) == 6588
    elif "zh_en" in dataset_name:
        assert len(dataset.instances) == 13245


