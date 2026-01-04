from abc import ABC, abstractmethod
import datasets
from pydantic import BaseModel
from tqdm import tqdm

class ScoredChunk(BaseModel):
    text: str
    index: int
    score: float

class AbsReranker(ABC):
    def __init__(self, n_to_rerank: int = 100, description : str = ""):
        self.n_to_rerank = n_to_rerank
        self.description = description
        

    @abstractmethod
    def rerank(self, query:str, chunks: list[str]) -> list[ScoredChunk]:
        """Reranks the chunks by order of relevance using the reranker

        Args:
            query (str): the query.
            chunks (list[str]): the list of chunks

        Returns:
            list[str]: the reranked list of chunks
        """

    def rerank_chunks_by_relevance_2d(
        self,
        ranked_chunks_idxes:list[list[int]],
        ranked_chunks_scores:list[list[int]],
        queries_dataset: datasets.Dataset,
        chunks_dataset:datasets.Dataset,
        n_to_rerank: int = 100
        ) -> tuple[list[list[int]], list[list[float]]]:
        """Considering the queries and chunks datasets, and considering the matrices
        of ordered chunk idx and scores returned by Retriever.rank_chunks_by_relevance_2d(),
        reranks the chunks by relevance regarding the queries.
        It returns results as 2D matrices of shape (len(queries), len(chunks)):
        - The first matrix corresponds to the indices of the chunks ranked by relevance regarding the query. 
        - The second matrix correspond to the relevance scores.

        Args:
            ranked_chunks_idxes (list[list[int]]): the idxes of chunks sorted by relevance for each query.
            ranked_chunks_scores (list[list[float]]): the scores of chunks sorted by relevance for each query.
            queries_dataset (datasets.Dataset): dataset of queries.
            chunks_dataset (datasets.Dataset): dataset of chunks.
            n_to_rerank (int): amout of best chunks to reranks.

        Returns:
            tuple[list[list[int]], list[list[float]]]: respecively the indices of the chunks
                sorted by relevance regarding each query, and the corresponding scores.
        """
        for i, query_sample in enumerate(tqdm(queries_dataset)):
            reranked_chunks = self.rerank(
                query_sample["query"],
                chunks_dataset[ranked_chunks_idxes[i][:n_to_rerank]]["text"] # only rerank top k
                )
            # remap idxes of chunks to their idx in dataset
            idxes_mapping = dict(enumerate(ranked_chunks_idxes[i][:n_to_rerank]))
            reranked_chunks_idxes = [idxes_mapping[c.index] for c in reranked_chunks]
            reranked_chunks_scores = [1 + c.score for c in reranked_chunks] # We add 1 to ensure the score of reranked chunks is always higher than score of cosine similarity and that sorting index won't mess order
            ranked_chunks_idxes[i][:n_to_rerank] = reranked_chunks_idxes
            ranked_chunks_scores[i][:n_to_rerank] = reranked_chunks_scores

        return ranked_chunks_idxes, ranked_chunks_scores
