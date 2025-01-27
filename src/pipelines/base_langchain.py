from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .abs_pipeline import AbsPipeline
from ..components import Chunk
from ..utils import dynamic_track_emissions


class BaseLangchainPipeline(AbsPipeline):
    """Basic pdf ingestion pipeline as suggested by langchain
    on their "getting started" section : https://python.langchain.com/docs/how_to/document_loader_pdf/
    """

    parsing_result: list[Document] | None

    @property
    def default_chunker(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

    @dynamic_track_emissions
    def _parse_file(self, filepath: str) -> str:
        """Parses a file.

        Args:
            filepath (str): path to a .pdf file.

        Returns:
            tuple[str, float]: the parsed string.
        """
        loader = PyPDFLoader(filepath)
        pages = list(loader.lazy_load())

        return pages

    def to_markdown(self, paginated_output: bool = False) -> str | dict[int, str]:
        if paginated_output:
            return dict(
                (i, page.page_content) for i, page in enumerate(self.parsing_result)
            )

        return "\n\n".join((page.page_content for page in self.parsing_result))

    @dynamic_track_emissions
    def _chunk_using_default_chunker(self) -> list[Document]:
        return self.default_chunker.create_documents(
            texts=[page.page_content for page in self.parsing_result],
            metadatas=[page.metadata for page in self.parsing_result],
        )

    def _process_default_chunker_output(self, chunks: list[Document]) -> list[Chunk]:
        """Formats the chunks object to Chunk object

        Args:
            chunks (list[ChunkNorrisChunk]): the raw chunks output by the package.

        Returns:
            list[Chunk]: the list of formatted chunks
        """
        return [
            Chunk(
                text=chunk.page_content,
                page_start=chunk.metadata["page"],
                page_end=chunk.metadata["page"],
                source_file=self.filename,
            )
            for chunk in chunks
        ]

    def _set_parser_with_device(self, device: Literal["cuda", "cpu"]):
        if device == "cuda":
            raise ValueError("BaseLangchainPipeline cannot involve usage of a gpu")
