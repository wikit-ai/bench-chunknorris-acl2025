import logging
from typing import Literal

from chunknorris.chunkers.tools import Chunk as ChunkNorrisChunk
from chunknorris.parsers.markdown.components import MarkdownDoc
from chunknorris.parsers import PdfParser
from chunknorris.chunkers import MarkdownChunker
from chunknorris.pipelines import PdfPipeline

from src.pipelines.abs_pipeline import AbsPipeline
from src.components import Chunk
from src.utils import dynamic_track_emissions

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)


class ChunkNorrisPipeline(AbsPipeline):
    """Uses chunknorris package"""

    parser: PdfParser
    parsing_result: MarkdownDoc | None

    @property
    def default_chunker(self) -> PdfPipeline:
        return PdfPipeline(self.parser, MarkdownChunker())

    # @dynamic_track_emissions
    def _parse_file(self, filepath: str) -> MarkdownDoc:
        """Parses a pdf file.

        Args:
            filepath (str): the path to a pdf file.

        Returns:
            MarkdownDoc: the output of the parser.
        """
        return self.parser.parse_file(filepath)

    def to_markdown(self, paginated_output: bool = False) -> str | dict[int, str]:
        return self.parser.to_markdown(keep_track_of_page=paginated_output)

    # @dynamic_track_emissions
    def _chunk_using_default_chunker(self) -> list[ChunkNorrisChunk]:
        """Get the chunks.

        Returns:
            tuple[list[Chunk], float]: returns the list of chunks,
                along with the latency to get them.
        """
        return self.default_chunker._get_chunks_using_strategy()

    def _process_default_chunker_output(
        self, chunks: list[ChunkNorrisChunk]
    ) -> list[Chunk]:
        """Formats the chunks from chunknorris's object to Chunk object

        Args:
            chunks (list[ChunkNorrisChunk]): the raw chunks output by the package.

        Returns:
            list[Chunk]: the list of formatted chunks
        """
        return [
            Chunk(
                text=chunk.get_text(),
                page_start=chunk.start_page,
                page_end=chunk.end_page,
                source_file=self.filename,
            )
            for chunk in chunks
        ]

    def _set_parser_with_device(self, device: Literal["cuda", "cpu"]) -> None:
        if device == "cuda":
            raise ValueError("ChunkNorris only runs on cpu")
        self.parser = PdfParser(use_ocr="auto" if self.use_ocr else "never")
