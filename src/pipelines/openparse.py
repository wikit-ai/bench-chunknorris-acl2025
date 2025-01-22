import os
from typing import Literal

import openparse
from openparse.schemas import Node
from openparse.doc_parser import ParsedDocument
from openparse import processing, DocumentParser

from dotenv import load_dotenv

from ..components import Chunk

load_dotenv()

from ..utils import timeit
from .abs_pipeline import AbsPipeline

class OpenParsePipeline(AbsPipeline):
    """Uses the OpenParse package : https://github.com/Filimoa/open-parse"""
    def __init__(
        self,
        chunking_type: Literal["basic", "semantic"],
        table_strategy:Literal["unitable", "pymupdf", "table-transformers"]
        ):
        super().__init__()

        self.use_semantic_chunking = chunking_type == "semantic"

        self.parser = DocumentParser(
            processing_pipeline=processing.NoOpIngestionPipeline(),
            table_args=OpenParsePipeline._get_table_args(table_strategy)
        )

    @property
    def default_chunker(self):
        if self.use_semantic_chunking:
            return processing.SemanticIngestionPipeline(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model="text-embedding-3-large",
                    min_tokens=64,
                    max_tokens=1024,
                )
        else:
            return processing.BasicIngestionPipeline()

    @timeit
    def parse_file(self, filepath: str) -> ParsedDocument:
        """Parses a file.

        Args:
            filepath (str): path to the file.

        Returns:
            tuple[list[Node], float]: parser output and latency.
        """
        self.parsing_result = self.parser.parse(filepath, ocr=False)
        return self.parsing_result
    
    def to_markdown(self):
        md_string = "\n\n".join((node.text for node in self.parsing_result.nodes))
        md_string = md_string.replace("<br>", "\n")
        return md_string

    @timeit
    def _chunk_using_default_chunker(self) -> list[Node]:
        """Openparse doesn't have a proper chunker so to say. But
        it has an interesting mechanic to merge the parsed nodes together.
        This pipeline is kind of part of the parsing but for the sake
        of comparison with other tools, we run the pipeline separatly."""
        return self.default_chunker.run(self.parsing_result.nodes)

    def _process_default_chunker_output(self, chunks: list[Node]) -> list[Chunk]:
        return [
            Chunk(
                text=node.text.replace("<br>", "\n"),
                page_start=node.bbox[0].page,
                page_end=node.bbox[0].page
            )
            for node in chunks
        ]

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
