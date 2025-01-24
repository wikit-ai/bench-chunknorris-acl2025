from typing import Literal

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from .abs_pipeline import AbsPipeline
from ..utils import dynamic_track_emissions

class MarkerPipeline(AbsPipeline):
    """Uses the Marker package : https://github.com/VikParuchuri/marker
    """
    def __init__(self, chunker = None, device = "gpu"):
        super().__init__(chunker, device)
        self.parser = PdfConverter(
            artifact_dict=create_model_dict(),
        )


    @property
    def default_chunker(self):
        return None

    @dynamic_track_emissions
    def _parse_file(self, filepath:str) -> str:
        """Parses a .pdf file.

        Args:
            filepath (str): path a a .pdf file.
        """
        rendered = self.parser(filepath)
        text, _, _ = text_from_rendered(rendered)

        return text

    @dynamic_track_emissions
    def _chunk_using_default_chunker(self):
        raise NotImplementedError()

    def _process_default_chunker_output(self, chunks):
        raise NotImplementedError()

    def _set_parser_using_device(self, device: Literal["cpu", "cuda"]):
        raise NotImplementedError("Does not apply to Marker")
