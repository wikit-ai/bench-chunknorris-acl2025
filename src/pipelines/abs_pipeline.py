from abc import ABC, abstractmethod
from typing import Any

from ..components import Chunk
from ..utils import timeit


class AbsPipeline(ABC):
    """The abstract pipeline from which each of the package's pipeline
    should be derived.
    """
    def __init__(self):
        pass

    
    @timeit # use this decorator so that the function returns the latency along with its output
    @abstractmethod
    def parse_file(self, filepath: str) -> Any:
        """Parses the file provided. Can return
        any type corresponding to the package used.
        //!\\ MUST ONLY contain code performing parsing, NOT chunking.

        Args:
            filepath (str): the path to the file

        Returns:
            Any: The return of the librairy's parser
        """

    @abstractmethod
    def to_markdown(self) -> str:
        """Returns a markdown string of the pdf's content
        """

    def chunk(self) -> tuple[list[Chunk], float]:
        """Chunks the parsed file using the package.
        Converts the package chunking's output to a list of Chunk
        objects. Also returns the latency obtained from the timeit decorator

        Returns:
            list[Chunk]: the list of chunks
            float: the latency = time used by the package to perform chunking.
        """
        raw_chunks, latency = self._chunk()
        chunks = self._process_chunking_output(raw_chunks)

        return chunks, latency


    @timeit # use this decorator so that the function returns the latency along with its output
    @abstractmethod
    def _chunk(self) -> Any:
        """Chunks the parsed file using the package and return the package's output "as is".
        //!\\ MUST ONLY contain the code performing chunking, NOT parsing.

        Returns:
            list[Chunk]: the list of chunks
            float: the latency = time used by the package to perform chunking.
        """

    @abstractmethod
    def _process_chunking_output(self, chunks: Any) -> list[Chunk]:
        """Formats the chunks returned by the chunking method of the package
        to a list of "Chunk" objects.

        Returns:
            list[Chunk]: the list of chunks
        """

    @abstractmethod
    def set_device(self, device: Any) -> None:
        """Sets the device to be used (gpu or cpu).

        Args:
            device (Any): the device to be set
        """
