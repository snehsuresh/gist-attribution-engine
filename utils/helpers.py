# utils/helpers.py
import os
import re

def sanitize_filename(text: str) -> str:
    """
    Converts a query or text into a filesystem-safe lowercase string
    by replacing non-alphanumeric characters with underscores.
    """
    return re.sub(r"[^\w\d]+", "_", text.lower()).strip("_")
