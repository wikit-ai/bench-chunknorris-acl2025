from typing import Any, Literal

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorDevice,
    AcceleratorOptions
    )
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.base import BaseChunk

from transformers import AutoTokenizer

from .abs_pipeline import AbsPipeline
from ..components import Chunk
from ..utils import dynamic_track_emissions

class DoclingPipeline(AbsPipeline):
    """Uses docling : https://github.com/DS4SD/docling"""
    parser = DocumentConverter

    def __init__(
            self,
            chunker = None,
            device = "gpu",
            tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            ):
        super().__init__(chunker, device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.parsing_result = None

    @property
    def default_chunker(self):
        return HybridChunker(
            tokenizer = self.tokenizer,
            )


    def _set_parser_with_device(self, device: Literal["cuda", "cpu"]):
        """Sets the parsers using specified device.

        Args:
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;]): the device to use.
        """
        match device:
            case "cpu":
                accelerator_options = AcceleratorOptions(
                    num_threads=4, device=AcceleratorDevice.CPU
                )
            case "cuda":
                accelerator_options = AcceleratorOptions(
                    num_threads=4, device=AcceleratorDevice.CUDA
                )
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            accelerator_options=accelerator_options
            )

        self.parser = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )


    @dynamic_track_emissions
    def _parse_file(self, filepath:str) -> Any:
        """Parses a file.

        Args:
            filepath (str): the path to the file.

        Returns:
            Any: The result of the parser.
        """
        return self.parser.convert(filepath)


    def to_markdown(self) -> str:
        """Get the parsed document as a markdown formatted string.

        Raises:
            RuntimeError: parse_file() must be called before running this

        Returns:
            str: the markdown string
        """
        return self.parsing_result.document.export_to_markdown()


    @dynamic_track_emissions
    def _chunk_using_default_chunker(self) -> list[BaseChunk]:
        """Gets the chunks.

        Returns:
            list[Chunk]: a list of chunks
        """
        return self.default_chunker.chunk(self.parsing_result.document)


    def _process_default_chunker_output(self, chunks:list[BaseChunk]) -> list[Chunk]:
        """Formats the chunks from docling's object to Chunk object

        Args:
            chunks (list[ChunkNorrisChunk]): the raw chunks output by the package.

        Returns:
            list[Chunk]: the list of formatted chunks
        """
        return [
            Chunk(
                text=self.default_chunker.serialize(chunk),
                page_start=DoclingPipeline._get_origin_page_of_chunk(chunk, "min"),
                page_end=DoclingPipeline._get_origin_page_of_chunk(chunk, "max"),
                source_file=self.filename
                )
            for chunk in chunks
        ]


    @staticmethod
    def _get_origin_page_of_chunk(chunk: BaseChunk, min_or_max: Literal["min", "max"]):
        """Get the source pages the chunk comes from.

        Args:
            chunk (BaseChunk): The chunk returned by DocLing.
            min_or_max (Literal["min", "max"]): whether we want to return
                the first of last page the chunkis from.

        Returns:
            list[int]: the list of all pages the chunk is sourced from.
        """
        match min_or_max:
            case "min":
                return min(
                    prov_item.page_no
                    for item in chunk.meta.doc_items
                    for prov_item in item.prov
                    )
            case "max":
                return max(
                    prov_item.page_no
                    for item in chunk.meta.doc_items
                    for prov_item in item.prov
                    )
