import re
from typing import Literal

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import MarkdownOutput
from marker.config.parser import ConfigParser

from .abs_pipeline import AbsPipeline
from ..utils import dynamic_track_emissions

class MarkerPipeline(AbsPipeline):
    """Uses the Marker package : https://github.com/VikParuchuri/marker
    """
    parser: PdfConverter
    parsing_result: MarkdownOutput

    @property
    def default_chunker(self):
        return None
    
    @property
    def marker_config(self):
        """the config for marker"""
        return {
            "output_format": "markdown",
            "force_ocr": False,
            "paginate_output": True,
            "disable_multiprocessing": True, # ? maybe not
            "disable_image_extraction": True,
        }


    def _set_parser_with_device(self, device: Literal["cpu", "cuda"]):
        if device == "cpu":
            raise ValueError("Marker doesn't allow to easily only run on cpu when GPU is available. Use an environment with pytorch-cpu to force not using cuda")

        config_parser = ConfigParser(self.marker_config)
        self.parser = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )


    @dynamic_track_emissions
    def _parse_file(self, filepath:str) -> MarkdownOutput:
        """Parses a .pdf file.

        Args:
            filepath (str): path a a .pdf file.
        """
        rendered = self.parser(filepath)

        return rendered

    def to_markdown(self, paginated_output = False) -> str | dict[int, str]:
        """Convert parsed output to markdown with optional pagination.
        
        If `keep_track_of_page` is True, returns a dictionary with page numbers as keys
        and page content as values. Otherwise, returns the markdown string with page 
        markers removed.

        Note : Marker, when used with its config "paginate_output": True inserts flag in the text :
        
        this is random text
        \n
        {12}-------------------
        \n
        some more text"

        with the number between brackets being the page.

        Returns:
            str | dict[int, str]: Markdown string or dictionary with paginated content.
        """
        pattern = r"\n{(\d+)}-{3,}\n"

        if paginated_output:
            matches = list(re.finditer(pattern, self.parsing_result.markdown))
            output = {}
            for i, match in enumerate(matches):
                page_content_start = match.end()
                page_content_end = matches[i + 1].start() if i + 1 < len(matches) else None
                page_number = int(match.group(1))
                page_content = self.parsing_result.markdown[page_content_start:page_content_end].strip()
                output[page_number] = page_content
            return output

        return re.sub(pattern, "\n", self.parsing_result.markdown)


    @dynamic_track_emissions
    def _chunk_using_default_chunker(self):
        raise NotImplementedError()

    def _process_default_chunker_output(self, chunks):
        raise NotImplementedError()

