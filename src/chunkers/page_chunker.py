from .abs_chunker import AbstractChunker
from ..components import Chunk


class PageChunker(AbstractChunker):
    """Simply returns a chunk per page of the pdf"""

    def chunk(self, paginated_md: dict[int, str], source_file: str) -> list[Chunk]:
        return [
            Chunk(
                text=md_string, page_start=page, page_end=page, source_file=source_file
            )
            for page, md_string in paginated_md.items()
        ]
