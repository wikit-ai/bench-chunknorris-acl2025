"""Utility functions"""

import os
import time
from typing import Any, Callable
from functools import wraps


def get_pdf_filepaths(directory:str) -> list[str]:
    """Considering a directory,
    get the filpath of every pdf file in it.

    Args:
        directory (str): the path to the directory

    Returns:
        list[str]: the list of absolute filepaths
    """
    pdf_filepaths: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_filepaths.append(os.path.abspath(os.path.join(root, file)))  
    
    return pdf_filepaths



def timeit(function: Callable[..., Any]) -> Any:
    """Meant to be used as a decorator using @timeit
    in order to measure the execution time of a function.

    Args:
        function (Callable[..., Any]): the function to measure exec time for.

    Returns:
        Any: the return of the function.
    """

    @wraps(function)
    def wrapper(*args: tuple[Any], **kwargs: dict[Any, Any]) -> tuple[Any, float]:
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()

        return result, end_time - start_time

    return wrapper
