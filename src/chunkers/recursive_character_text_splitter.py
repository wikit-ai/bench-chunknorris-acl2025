from langchain_text_splitters import RecursiveCharacterTextSplitter

from .abs_chunker import AbstractChunker

class RecursiveCharacterChunker(AbstractChunker):
    """Simple wrapper around langchain's RecursiveCharacterTextSplitter
    to ensure in fits in the pipelines"""
    chunker : RecursiveCharacterTextSplitter

    def __init__(self):
        self.chunker = RecursiveCharacterTextSplitter(
            toto
        )
