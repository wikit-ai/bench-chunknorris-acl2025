from abc import ABC, abstractmethod
import os
from typing import Any, Literal
from torch.cuda import is_available
from src.components import Chunk
from src.chunkers.abs_chunker import AbstractChunker
from src.utils import dynamic_track_emissions



class AbsPipeline(ABC):
    """The abstract pipeline from which each of the package's pipeline
    should be derived.
    """

    parsing_result: Any | None = (
        None  # used to store the parsing result in memory to perform chunking separatly
    )
    device: Literal["gpu", "cpu"] = "cpu"
    filename: str | None = None  # store the name of last filename parsed.

    def __init__(
        self,
        chunker: AbstractChunker | None = None,
        device: Literal["gpu", "cpu"] = "cpu",
        use_ocr: bool = False,
    ):
        self.external_chunker = chunker
        self.use_ocr = use_ocr
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
    def parse_files(self, filepaths: list[str]) -> None:
        """Parses a list of files and measures the energy consumption.

        Args:
            filepaths (list[str]): a list of files to parse

        """
        _ = [self.parse_file(filepath) for filepath in filepaths]


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
    def to_markdown(self, paginated_output: bool = False) -> str | dict[int, str]:
        """Convert parsed output to markdown with optional pagination.

        If `paginated_output` is True, returns a dictionary with page numbers (0-based !) as keys
        and page content as values. Otherwise, returns the markdown string with page
        markers removed.
        """

    def chunk(self) -> list[Chunk]:
        """Chunks the parsed file using the package.
        Converts the package chunking's output to a list of Chunk objects.

        Returns:
            list[Chunk]: the list of chunks
            float: the latency = time used by the package to perform chunking.
        """
        if self.external_chunker is not None:
            paginated_md = self.to_markdown(paginated_output=True)
            chunks = self.external_chunker.chunk(paginated_md, self.filename)
        else:
            raw_chunks = self._chunk_using_default_chunker()
            chunks = self._process_default_chunker_output(raw_chunks)

        return chunks

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
    def _set_parser_with_device(self, device: Literal["cuda", "cpu"]):
        """Set the parser so that it uses the specified device.
        --> This is where self.parser may be set

        Args:
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;]): the device to be used
        """

    def set_device(self, device: Literal["cuda", "cpu"]) -> None:
        """Sets the device to be used (gpu or cpu).

        Args:
            device (Any): the device to be set
        """
        device = device if is_available() else "cpu"
        print(f"Running pipeline on {device}")
        self._set_parser_with_device(device)
        self.device = device
