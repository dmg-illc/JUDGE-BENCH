"""
Utility and helper functions
"""

from typing import Any
import json


def read_jsonl(file_path: str) -> list[dict[str, Any]]:
    """
    Reads a JSONL file and returns a list of dictionaries.

    Each line in the JSONL file should be a valid JSON object. The function
    will attempt to parse each line and collect them into a list.

    Parameters:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
      represents a JSON object from the file.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        JSONDecodeError: If a line in the file is not valid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [json.loads(line.strip()) for line in file if line.strip()]
    except FileNotFoundError:
        print(
            f"Error: The file at {file_path} was not found. If you haven't, please download the original data using `download_original_data.sh`."
        )
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        raise


def save_dict_to_json(data: dict[str, Any], file_path: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Parameters:
        data (Dict[str, Any]): The dictionary to save.
        file_path (str): The path to the file where the dictionary should be saved.

    Raises:
        ValueError: If the provided file_path is not valid.
        IOError: If an I/O error occurs during file writing.

    Returns:
        None

    Example:
        >>> my_data = {'key': 'value', 'numbers': [1, 2, 3]}
        >>> save_dict_to_json(my_data, 'data.json')
    """
    # Check if the file_path provided is valid
    if not file_path.endswith(".json"):
        raise ValueError("File path must end with '.json'")

    # Write the dictionary to a JSON file
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError as e:
        raise IOError(f"An error occurred while writing to the file: {e}")
