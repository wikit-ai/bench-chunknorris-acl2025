from abc import ABC, abstractmethod
import os
from typing import Any, Literal

from dotenv import load_dotenv

from ..components import Chunk
from ..chunkers import AbstractChunker
from ..utils import dynamic_track_emissions

load_dotenv()

class AbsPipeline(ABC):
    """The abstract pipeline from which each of the package's pipeline
    should be derived.
    """
    parsing_result: Any | None = None # used to store the parsing result in memory to perform chunking separatly
    device: Literal["gpu", "cpu"] = "cpu"
    filename: str | None = None # store the name of last filename parsed.

    def __init__(self, chunker: AbstractChunker | None = None, device: Literal["gpu", "cpu"] = "cpu"):
        self.external_chunker = chunker
        self.set_device(device)

    @property
    @abstractmethod
    def default_chunker(self) -> Any | None:
        """If the package also includes a chunker,
        this returns the chunker, else returns None"""


    def parse_file(self, filepath: str) -> None:
        """Parses the file.

        Args:
            filepath (str): the path to the pdf file.

        Returns:
            Any: _description_
        """
        self.parsing_result = self._parse_file(filepath)
        self.filename = os.path.basename(filepath)


    @dynamic_track_emissions
    @abstractmethod
    def _parse_file(self, filepath: str) -> Any:
        """Parses the file provided.
        - Can return any type corresponding to the package used
        - output is stored in self.parsing_result for later use
        - //!\\ MUST ONLY contain code performing parsing, NOT chunking
        - //!\\ MUST be decorated with @dynamic_track_emissions

        Args:
            filepath (str): the path to the file

        Returns:
            Any: The return of the librairy's parser
        """

    @abstractmethod
    def to_markdown(self) -> str:
        """Returns a markdown string of the parsed pdf's content
        """

    def chunk(self) -> tuple[list[Chunk], float]:
        """Chunks the parsed file using the package.
        Converts the package chunking's output to a list of Chunk
        objects. Also returns the latency obtained from the timeit decorator

        Returns:
            list[Chunk]: the list of chunks
            float: the latency = time used by the package to perform chunking.
        """
        if self.external_chunker is not None:
            md_string = self.to_markdown()
            chunks, latency = self.external_chunker.chunk(md_string)
        else:
            raw_chunks, latency = self._chunk_using_default_chunker()
            chunks = self._process_default_chunker_output(raw_chunks)

        return chunks, latency


    @dynamic_track_emissions
    @abstractmethod
    def _chunk_using_default_chunker(self) -> Any:
        """Chunks the parsed file using the chunker provided by the package
        and return the package's output "as is".
        //!\\ MUST ONLY contain the code performing chunking, NOT parsing.
        //!\\ MUST be decorated with @dynamic_track_emissions

        Returns:
            list[Chunk]: the list of chunks
            float: the latency = time used by the package to perform chunking.
        """

    @abstractmethod
    def _process_default_chunker_output(self, chunks: Any) -> list[Chunk]:
        """Formats the chunks returned by the chunking method of the package
        to a list of "Chunk" objects.

        Returns:
            list[Chunk]: the list of chunks
        """

    @abstractmethod
    def _set_parser_with_device(self, device : Literal["cuda", "cpu"]):
        """Set the parser so tat it uses the specified device.
        This is where self.parser may be set

        Args:
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;]): the device to be used
        """


    def set_device(self, device: Literal["cuda", "cpu"]) -> None:
        """Sets the device to be used (gpu or cpu).

        Args:
            device (Any): the device to be set
        """
        self._set_parser_with_device(device)
        self.device = device