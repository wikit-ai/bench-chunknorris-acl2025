from pydantic import BaseModel, Field

class Chunk(BaseModel):
    text: str = Field(description="The chunk's text.")
    page_start: int = Field(description="The chunk's starting page regarding its source pdf file, 0-based.")
    page_end: int = Field(description="The chunk's ending page regarding its source pdf file, 0-based.")
