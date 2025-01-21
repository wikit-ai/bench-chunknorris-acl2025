import time

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from .abs_pipeline import AbsPipeline
from ..utils import timeit

class MarkerPipeline(AbsPipeline):
    def __init__(self):
        self.parser = PdfConverter(
            artifact_dict=create_model_dict(),
        )

        self.parsing_result = None

    @timeit
    def parse_file(self, filepath:str) -> str:
        """Parses a .pdf file.

        Args:
            filepath (str): path a a .pdf file.
        """
        rendered = self.parser(filepath)
        text, _, _ = text_from_rendered(rendered)

        self.parsing_result = text

        return text

    @timeit
    def _chunk(self):
        raise NotImplementedError()