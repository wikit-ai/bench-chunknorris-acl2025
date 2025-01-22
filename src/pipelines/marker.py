from typing import Literal

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from .abs_pipeline import AbsPipeline
from ..utils import timeit

class MarkerPipeline(AbsPipeline):
    """Uses the Marker package : https://github.com/VikParuchuri/marker
    """
    def __init__(self):
        super().__init__()
        self.parser = PdfConverter(
            artifact_dict=create_model_dict(),
        )


    @property
    def default_chunker(self):
        return None

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
    def _chunk_using_default_chunker(self):
        raise NotImplementedError()

    def _process_default_chunker_output(self, chunks):
        raise NotImplementedError()

    def set_device(self, device: Literal["cpu", "cuda"]):
        raise NotImplementedError("Does not apply to Marker")
