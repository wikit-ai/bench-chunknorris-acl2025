from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represent a chunk of a document."""

    text: str = Field(description="The chunk's text.")
    page_start: int = Field(
        description="The chunk's starting page regarding its source pdf file, 0-based."
    )
    page_end: int = Field(
        description="The chunk's ending page regarding its source pdf file, 0-based."
    )
    source_file: str = Field(
        description="The name of the file the chunk originated from"
    )


class ReferenceSection(BaseModel):
    source_file: str = Field(
        description="The name of the file the reference passage originated from."
    )
    query: str = Field(description="The user's query.")
    target_page: int = Field(
        description="The target_passage page regarding its source pdf file, 0-based."
    )
    target_passage: str = Field(
        description="The reference passage that makes it possible to answer the user's question."
    )


class GeneratedAnswer(BaseModel):
    llm: str = Field(description="The name of the LLM used to generate the answer.")
    generation: str = Field(description="The text of the generated answer.")
