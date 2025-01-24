from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    top_k: int = Field(description="The top k chunks to consider.")
    mrr: float = Field(description="The Mean Reciprocal Rank og the pipeline.")
    recall: float = Field(description="The Recall of the pipeline.")
    to_generate: list[bool] = Field(
        description="""A list indicating whether the chunk containing the answer has been retrieved 
        and if so, the answer can be generated."""
    )
    index_presence_chunks: list[int | None] = Field(
        description="""A list of indices indicating the presence of chunks containing the answer,
          or None if not present."""
    )
