from abc import ABC, abstractmethod

from ..components import Chunk

class AbstractChunker(ABC):
    """Intended to be used as a base class for all
    chunker that are being used in pipelines"""
    def __init__(self):
        pass

    @abstractmethod
    def chunk(self, md_string: str) -> list[Chunk]:
        """Chunks the provided string.

        Args:
            md_string (str): the markdown string to chunk.

        Returns:
            list[Chunk]: a list of Chunk objects
        """
