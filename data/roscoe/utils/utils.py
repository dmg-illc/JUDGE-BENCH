"""
Utility and helper functions
"""

from typing import Any
import json
import re
import pandas as pd
from pandas import DataFrame


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


def read_csv(file_path: str) -> DataFrame:
    """
    Reads csv file and stores it in a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        Optional[list]: The list of column names if the file exists and is a valid CSV, otherwise None.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file cannot be parsed.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No file found at the specified path: {file_path}. If you haven't, please download the original data using `download_original_data.sh`."
        )
    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")
    except pd.errors.ParserError:
        print("Failed to parse the CSV file. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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


def normalize_string(s: str) -> str:
    """
    Normalizes a string by converting it to lowercase and removing all non-alphanumeric characters.

    Parameters:
        s (str): The input string to be normalized.

    Returns:
        str: The normalized string with non-alphanumeric characters removed.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def compare_strings(s1: str, s2: str) -> bool:
    """
    Compare two strings for equality after normalizing them.

    Parameters:
        s1 (str): The first string to compare.
        s2 (str): The second string to compare.

    Returns:
        bool: True if the normalized versions of the strings are equal, False otherwise.
    """
    normalized_s1 = normalize_string(s1)
    normalized_s2 = normalize_string(s2)

    return normalized_s1 == normalized_s2


def parse_reasoning_chain(text: str) -> str:
    """
    Removes specific patterns from the input string, particularly the patterns
    that start with '<br />&nbsp&nbspStep' followed by a number and a dash.

    Parameters:
        text (str): The input text from which to remove the patterns.

    Returns:
        str: The text with the specified patterns removed.
    """
    base_pattern = r"<br\s*/>"
    refined_text = re.sub(base_pattern, "", text)
    pattern = r"&nbsp&nbspStep\s+\d+\s*-"
    return re.sub(pattern, "", refined_text)


def split_substeps(text: str) -> list[str]:
    """
    Splits the input text into substeps and returns a list of strings.

    Parameters:
    - text (str): The input text to be split into substeps.

    Returns:
    - List[str]: A list of strings representing the substeps after splitting.
    """
    return text.split("<br />&nbsp&nbsp")[1:]
