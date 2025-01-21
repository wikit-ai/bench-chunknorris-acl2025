import time

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


class MarkerPipeline():
    def __init__(self):
        self.parser = PdfConverter(
            artifact_dict=create_model_dict(),
        )

        self.parsing_result = None

    def parse_file(self, filepath:str):
        """Parses a .pdf file.

        Args:
            filepath (str): path a a .pdf file.
        """
        rendered = self.parser(filepath)
        start_time = time.perf_counter()
        text, _, _ = text_from_rendered(rendered)
        end_time = time.perf_counter()

        self.parsing_result = text

        return text, end_time - start_time

    def chunk(self):
        raise NotImplementedError()