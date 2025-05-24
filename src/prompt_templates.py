def load_prompt(path: str = 'chatbot_prompt.txt') -> str:
    """Loads the prompt from a text file.

    Args:
        path (str): The path to the text file containing the prompt.

    Returns:
        str: The content of the prompt file as a single string.

    Raises:
        FileNotFoundError: If the file specified by 'path' is not found.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at path: {path}")