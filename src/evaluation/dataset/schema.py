
from pydantic import BaseModel


class ReferenceSection(BaseModel):
    source_file: str
    query: str
    page: int
    passage: str
