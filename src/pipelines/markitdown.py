from markitdown import MarkItDown

from ..utils import timeit
from .abs_pipeline import AbsPipeline

class MarkitDownPipeline(AbsPipeline):
    def __init__(self):
        self.parser = MarkItDown()
        self.parsing_result = None

    @timeit
    def parse_file(self, filepath:str) -> str:
        """Parses a file.

        Args:
            filepath (str): path to a .pdf file.

        Returns:
            tuple[str, float]: the parsed string.
        """
        self.parsing_result = self.parser.convert(filepath)

        return self.parsing_result.text_content


    def _chunk(self):
        raise NotImplementedError()


    def set_device(self, device):
        raise NotImplementedError()

