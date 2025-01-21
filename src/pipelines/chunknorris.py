import logging


from chunknorris.chunkers.tools import Chunk as ChunkNorrisChunk
from chunknorris.parsers.markdown.components import MarkdownDoc
from chunknorris.parsers import PdfParser
from chunknorris.chunkers import MarkdownChunker
from chunknorris.pipelines import PdfPipeline

from .abs_pipeline import AbsPipeline
from ..components import Chunk
from ..utils import timeit

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)

class ChunkNorrisPipeline(AbsPipeline):
    """Uses chunknorris"""
    def __init__(self):
        self.parser = PdfParser(use_ocr="never")
        self.chunker = MarkdownChunker()
        self.pipeline = PdfPipeline(self.parser, self.chunker)

        self.parsing_result = None

    @timeit
    def parse_file(self, filepath:str) -> MarkdownDoc:
        """Parses a pdf file.

        Args:
            filepath (str): the path to a pdf file.

        Returns:
            MarkdownDoc: the output of the parser.
        """
        self.parsing_result = self.pipeline.parser.parse_file(filepath)

        return self.parsing_result
    
    def to_markdown(self):
        return self.parser.to_markdown()

    @timeit
    def _chunk(self) -> list[ChunkNorrisChunk]:
        """Get the chunks.

        Returns:
            tuple[list[Chunk], float]: returns the list of chunks,
                along with the latency to get them.
        """
        return self.pipeline._get_chunks_using_strategy()


    def _process_chunking_output(self, chunks :list[ChunkNorrisChunk]) -> list[Chunk]:
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
                page_end=chunk.end_page
                )
            for chunk in chunks
        ]
    

    def set_device(self):
        """Doesn't apply to chunknorris"""
        pass
