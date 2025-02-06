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


class RagItem(BaseModel):
    reference_section: ReferenceSection = Field(
        description="The reference section associated with the RAG item."
    )
    top_chunks: list[str] = Field(
        description="A list of top chunks retrieved for the RAG item."
    )
    correct_chunk: int = Field(
        description="The index of the correct chunk within the top chunks."
    )
