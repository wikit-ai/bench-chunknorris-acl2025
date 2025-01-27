from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.components import Chunk
from src.chunkers.abs_chunker import AbstractChunker


class RecursiveCharacterChunker(AbstractChunker):
    """Simple wrapper around langchain's RecursiveCharacterTextSplitter
    to ensure in fits in the pipelines"""

    chunker: RecursiveCharacterTextSplitter

    def __init__(self, *args, **kwargs):
        self.chunker = RecursiveCharacterTextSplitter(*args, **kwargs)

    def chunk(self, paginated_md, source_file):
        raw_chunks = self.chunker.create_documents(
            texts=list(paginated_md.values()),
            metadatas=[{"page": key} for key in paginated_md.keys()],
        )

        chunks = [
            Chunk(
                text=chunk.page_content,
                page_start=chunk.metadata["page"],
                page_end=chunk.metadata["page"],
                source_file=source_file,
            )
            for chunk in raw_chunks
        ]

        return chunks
