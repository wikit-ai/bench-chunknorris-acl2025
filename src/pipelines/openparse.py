from typing import Literal
import time

import openparse
from openparse.schemas import Node
from openparse import processing, DocumentParser

from ..utils import timeit
from .abs_pipeline import AbsPipeline

class OpenParsePipeline(AbsPipeline):
    def __init__(
        self,
        table_strategy:Literal["unitable", "pymupdf", "table-transformers"]
        ):

        self.pipeline = processing.BasicIngestionPipeline()
        self.parser = DocumentParser(
            processing_pipeline=self.pipeline,
            table_args=OpenParsePipeline._get_table_args(table_strategy)
        )
        self.parsed_result = None

    @timeit
    def parse_file(self, filepath: str) -> list[Node]:
        """Parses a file.

        Args:
            filepath (str): path to the file.

        Returns:
            tuple[list[Node], float]: parser output and latency.
        """
        self.parsed_result = self.parser.parse(filepath).nodes

        return self.parsed_result


    def _chunk(self):
        raise NotImplementedError()


    @staticmethod
    def _get_table_args(
        table_strategy:Literal["unitable", "pymupdf", "table-transformers"]
        ):
        match table_strategy:
            case "unitable":
                return {
                    "parsing_algorithm": "unitable",
                    "table_output_format": "html"
                }
            case "pymupdf":
                return {
                    "parsing_algorithm": "pymupdf",
                    "table_output_format": "markdown"
                }
            case "table-transformers":
                return {
                    "parsing_algorithm": "table-transformers",
                    "table_output_format": "markdown"
                }


    def set_device(self, device:Literal["cuda", "cpu"]):
        openparse.config.set_device(device)
