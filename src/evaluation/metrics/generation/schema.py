from pydantic import BaseModel, Field

from src.components import ReferenceSection


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


class GenerationItem(BaseModel):
    name_pipeline: str = Field(
        description="The name of the pipeline used for generation."
    )
    correct_chunk_index: int = Field(
        description="The index of the correct chunk used for generation."
    )
    answer: str = Field(description="The generated answer.")
    input_tokens: int = Field(
        description="The number of input tokens used for generation."
    )
    output_tokens: int = Field(description="The number of output tokens generated.")
    energy_min: float = Field(
        description="The minimum energy impact of the generation."
    )
    energy_max: float = Field(
        description="The maximum energy impact of the generation."
    )
    gwp_min: float = Field(
        description="The minimum global warming potential impact of the generation."
    )
    gwp_max: float = Field(
        description="The maximum global warming potential impact of the generation."
    )
    price: float = Field(description="The cost of the generation.")


class GenerationCollection(BaseModel):
    reference: ReferenceSection = Field(
        description="The reference section associated with the generated response."
    )
    generated_response: GenerationItem = Field(description="The generated response.")
