import json
import os

def save_dict(filename, new_scores):
    """
    Save or update a dictionary in a JSON file.

    If the file exists and contains valid JSON, it will be loaded and updated
    with the new key-value pairs from `new_scores`. Otherwise, a new file will be created.

    Parameters:
        filename (str): Path to the JSON file.
        new_scores (dict): Dictionary containing new key-value pairs to save.

    Returns:
        None
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                scores = json.load(f)
        except (json.JSONDecodeError, IOError):
            scores = {}
    else:
        scores = {}

    scores.update(new_scores)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=4)

def load_dict(filename):
    """
    Load a dictionary from a JSON file.

    If the file does not exist or contains invalid JSON,
    an empty dictionary is returned and an error message is printed.

    Parameters:
        filename (str): Path to the JSON file.

    Returns:
        dict: Dictionary loaded from the file, or an empty dict on error.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Error] File not found: '{filename}'")
    except json.JSONDecodeError:
        print(f"[Error] Invalid JSON in file: '{filename}'")
    except IOError:
        print(f"[Error] Could not read file: '{filename}'")
    
    return {}
