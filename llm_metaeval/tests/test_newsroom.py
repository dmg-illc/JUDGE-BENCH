"""
Test newsroom dataset against pre-defined data schema
"""

import os
import json
import pytest
from typing import Any
from pydantic import ValidationError

from llm_metaeval.data.dataclasses import Dataset

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(
    os.path.join(FILE_DIRECTORY, os.pardir, os.pardir, "data", "newsroom")
)


@pytest.fixture
def load_json_data(request) -> dict[str, Any]:
    """
    A fixture to load JSON data from a specified file.

    Args:
        request: The fixture request object that contains params.

    Returns:
        dict[str, Any]: The JSON data loaded from the specified file.

    Raises:
        FileNotFoundError: If the JSON file cannot be found.
        json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    file_name = request.param
    file_path = os.path.join(DATA_FOLDER, file_name)
    with open(file_path, "r") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "load_json_data, expected",
    [
        (
            "newsroom.json",
            {"instances": 420, "annotations": 4, "first_id": 1},
        ),
    ],
    indirect=["load_json_data"],
)
def test_datasets(load_json_data, expected):
    """
    Test to ensure converted data loaded adheres to the data schema.

    Args:
        load_data (Dict[str, Any]): The JSON data loaded from a file.
        expected (Dict[str, int]): Expected values for the dataset properties.

    Raises:
        ValidationError: If the data does not conform to the Pydantic model.
    """
    try:
        dataset = Dataset.model_validate(load_json_data)
        assert dataset is not None, "Model validation failed; no data returned."
    except ValidationError as e:
        pytest.fail(f"Validation failed with errors: {e}")

    # Assertions based on the expected dictionary passed in
    assert len(dataset.instances) == expected["instances"], "Instance count mismatch."
    assert (
        len(dataset.annotations) == expected["annotations"]
    ), "Annotation count mismatch."
    assert (
        dataset.instances[0].id == expected["first_id"]
    ), "First instance ID mismatch."