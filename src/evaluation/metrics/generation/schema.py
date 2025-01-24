from pydantic import BaseModel, Field


class GenerationMetrics(BaseModel):
    rouge_precision: float = Field(
        description="The ROUGE precision score between generated text and reference text."
    )
    semantic_similarity: float = Field(
        description="The semantic similarity score between generated text and reference text."
    )
    avg_token_counts: float = Field(
        description="The average number of tokens in the generated text."
    )
