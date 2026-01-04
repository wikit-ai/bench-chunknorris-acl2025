import os
import logging
from litellm import rerank
from dotenv import load_dotenv

from src.rerankers.abs_reranker import AbsReranker, ScoredChunk


logger = logging.getLogger('LiteLLM')
logger.setLevel(logging.WARNING)
load_dotenv()

class CohereReranker(AbsReranker):
    """Uses Cohere reranker models"""
    def __init__(self, model_name: str = "rerank-v3.5", n_to_rerank : int = 100, description : str = ""):
        super().__init__(n_to_rerank=n_to_rerank, description=description)
        self.model_name = model_name
        assert os.getenv("COHERE_API_KEY", None) is not None, "Missing COHERE_API_KEY as env variable."

    def rerank(self, query:str, chunks: list[str]) -> list[ScoredChunk]:
        """
        Args:
            query (str): the query.
            chunks (list[str]): the list of chunks to rerank.

        Returns:
            list[ScoredChunk]: the reranked chunks, ranked by descreasing score.
        """
        results = rerank(
            model=f"cohere/{self.model_name}",
            query=query,
            documents=chunks,
            )

        return sorted([
            ScoredChunk(
                text=chunks[res["index"]],
                index=res["index"],
                score=res["relevance_score"],
            ) for res in results.results
        ], key=lambda x: x.score, reverse=True)
