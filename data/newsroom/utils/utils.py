"""
Utility and helper functions
"""

from typing import Any, Optional
import json
import pandas as pd


def read_csv_to_dataframe(
    file_path: str, delimiter: str = ",", encoding: str = "utf-8"
) -> Optional[pd.DataFrame]:
    """
    Reads a CSV file and stores the result in a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        delimiter (str, optional): The delimiter used in the CSV file. Default is ','.
        encoding (str, optional): The encoding of the CSV file. Default is 'utf-8'.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the CSV data, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except pd.errors.ParserError:
        print(
            f"Error: The file at '{file_path}' could not be parsed. Please check the delimiter."
        )
    except UnicodeDecodeError:
        print(
            f"Error: The file at '{file_path}' could not be decoded with encoding '{encoding}'."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def save_dict_to_json(data: dict[str, Any], file_path: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Args:
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
