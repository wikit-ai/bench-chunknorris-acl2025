"""Utility functions"""

import os
import time
from typing import Any, Callable
from functools import wraps

from codecarbon import track_emissions
from dotenv import load_dotenv

load_dotenv()


def get_pdf_filepaths(directory: str) -> list[str]:
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


def get_pipeline(pipeline_name: str):
    """Gets the appropriate pipeline, depending
    on the name specified in PACKAGE_NAME environment variable

    Raises:
        ValueError: If PACKAGE_NAME is not specified, or if name is not recognized

    Returns:
        AbsPipeline: the pipeline
    """
    match pipeline_name:
        case "base":
            from src.pipelines.base_langchain import BaseLangchainPipeline

            pipe = BaseLangchainPipeline()
        case "chunknorris":
            from src.pipelines.chunknorris import ChunkNorrisPipeline

            pipe = ChunkNorrisPipeline()
        case "docling":
            from src.pipelines.docling import DoclingPipeline

            pipe = DoclingPipeline()
        case "marker":
            from src.pipelines.marker import MarkerPipeline

            pipe = MarkerPipeline()
        case "openparse":
            from src.pipelines.openparse import OpenParsePipeline

            pipe = OpenParsePipeline()
        case None:
            raise ValueError("Missing environment variable 'PACKAGE_NAME'")
        case other:
            raise ValueError(f"'{other}' not recognized as a package name available")

    return pipe


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


def dynamic_track_emissions(func):
    """Wrapper of the track_emission decorator so that is has
    access to class state when called"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        experiment_id = "___".join(
            (
                self.__class__.__name__,
                func.__name__,
                self.__dict__.get("device", ""),
                self.__dict__.get("filename", ""),
            )
        )
        carbon_decorator = track_emissions(
            offline=True,
            experiment_id=experiment_id,
            country_iso_code=os.getenv("COUNTRY_ISO_CODE", "USA"),
        )

        return carbon_decorator(func)(self, *args, **kwargs)

    return wrapper
