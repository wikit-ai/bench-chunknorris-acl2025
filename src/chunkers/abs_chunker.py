from abc import ABC, abstractmethod

from ..components import Chunk

class AbstractChunker(ABC):
    """Intended to be used as a base class for all
    chunker that are being used in pipelines"""
    def __init__(self):
        pass

    @abstractmethod
    def chunk(self, paginated_md: dict[int, str], source_file: str) -> list[Chunk]:
        """Chunks the provided string.

        Args:
            paginated_md (dict[int, str]): a dict of Markdown formatted strings per page.
            source_file (str): the name of the file from which the paginated md originated.

        Returns:
            list[Chunk]: a list of Chunk objects.
        """
