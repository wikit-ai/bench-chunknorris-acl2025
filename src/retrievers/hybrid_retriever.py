from collections import defaultdict
from typing import Callable, Literal
import datasets
from datasets.features.features import Value
import numpy as np
from src.retrievers.abs_retriever import AbsRetriever

class HybridRetriever(AbsRetriever):
    """A basic dense retriever. It uses embedding and cosine similarity to retrieve most relevant chunks"""
    def __init__(
            self,
            retrievers: list[AbsRetriever],
            default_k : int = 10,
            queries_dataset: datasets.Dataset | None = None,
            chunks_dataset: datasets.Dataset | None = None,
            score_fusion_method: Literal["rrf", "minmax_rescaling"] = "rrf",
            description : str = ""
            ):
        """Initialises a dense retriever. Dense retriever uses embedding

        Args:
            retrievers (list[AbsRetriever]): The retrievers to use to retrieve chunks.
            default_k (int): the amount of chunks to retrieve by default when calling to retriever. Defaults to 10.
            queries_dataset (datasets.Dataset | None): the dataset of queries. Can be set to None if you plan to provided them later.
                NOTE: Must have a column "query" of type str which contains the queries.
            chunks_dataset (datasets.Dataset | None): the dataset which contains the chunks. Can be set to None if you plan to provide them later.
                NOTE: Must have a column "text" of type str which contains the text of the chunks.
            description (str |None): an additional description of the retriever. Mainly used to
                keep tracks of which retriever is which during evaluation. Defaults to None.
        """
        self.retrievers = retrievers
        self.score_fusion_method = score_fusion_method
        super().__init__(default_k=default_k, queries_dataset=queries_dataset, chunks_dataset=chunks_dataset, description=description)

    @property
    def score_rescaling_function(self) -> Callable[[list[float]], list[float]]:
        """Returns the function used to rescale the scores,
        based on the chosen score fusion method."""
        return {
            "rrf": HybridRetriever.rescale_scores_with_rrf,
            "minmax_rescaling": HybridRetriever.rescale_scores_with_min_max,
        }[self.score_fusion_method]

    @property
    def queries_dataset(self):
        """The dataset of queries"""
        if self._queries_dataset is None:
            raise ValueError("The queries_dataset must be set before accessing it.")
        return self._queries_dataset

    @queries_dataset.setter
    def queries_dataset(self, dataset: datasets.Dataset):
        if dataset is not None:
            HybridRetriever._validate_dataset_schema(dataset, {"query": Value("string")})
            # NOTE : This implies duplicating the dataset for each retriever. This may consume a lot of memory ! Suboptimal
            for retriever in self.retrievers:
                retriever.queries_dataset = dataset
        self._queries_dataset = dataset

    @property
    def chunks_dataset(self):
        """The dataset of chunks"""
        if self._chunks_dataset is None:
            raise ValueError("The chunks_dataset must be set before accessing it.")
        return self._chunks_dataset

    @chunks_dataset.setter
    def chunks_dataset(self, dataset: datasets.Dataset):
        if dataset is not None:
            HybridRetriever._validate_dataset_schema(dataset, {"text": Value("string")})
            dataset = dataset.add_column("index", range(len(dataset)))
            for retriever in self.retrievers:
                retriever.chunks_dataset = dataset
        self._chunks_dataset = dataset


    def retrieve_chunks(
        self, query: str, k: int | None = None
    ) -> datasets.Dataset:
        """Considering a query, retrieves the k most relevant chunks.
        Both query_sample and "chunks_dataset" must have a "emb" column which is the embedding.

        Args:
            query (str): the query to retrieve chunks for, as a string.
            k (int): the amount of chunks to retrieve. If None, uses the default_k. Defaults to None.

        Returns:
            dataset.Dataset: a dataset where each sample a chunk retrieved.
        """
        k = k or self.default_k
        idx_score_tuples = [
            (index, norm_score)
            for tops in [
                retriever.retrieve_chunks(query, k) for retriever in self.retrievers
                ]
            for index, norm_score in zip(tops["index"], self.score_rescaling_function(tops["score"]))
        ]
        idx_score_mapping = defaultdict(float)
        for index, score in idx_score_tuples:
            idx_score_mapping[index] += score

        topk_indices, topk_scores = list(zip(*sorted(
            idx_score_mapping.items(), key=lambda x: x[1], reverse=True
            )))
        top_chunks = self.chunks_dataset.select(topk_indices[:k])
        top_chunks = top_chunks.add_column("score", topk_scores[:k])

        return top_chunks

    def rank_chunks_by_relevance_2d(self) -> tuple[list[list[int]], list[list[float]]]:
        """Considering the queries and chunks datasets,
        ranks the chunks by relevance regarding the queries.
        It returns results as 2D matrices of shape (len(queries), len(chunks)):
        - The first matrix corresponds to the indices of the chunks ranked by relevance regarding the query. 
        - The second matrix correspond to the relevance scores.

        Returns:
            tuple[list[list[int]], list[list[float]]]: respecively the indices of the chunks
                sorted by relevance regarding each query, and the corresponding scores.
        """
        sorted_scores_all : list[np.array] = []
        for retriever in self.retrievers:
            ranked_indices, ranked_scores = retriever.rank_chunks_by_relevance_2d()
            ranked_scores = np.array([self.score_rescaling_function(scores) for scores in ranked_scores])
            # Reorder the scores by indices instead of score value so that we can add up the scores of each chunk
            sorted_indices = np.argsort(ranked_indices, axis=1)
            sorted_scores = np.take_along_axis(ranked_scores, sorted_indices, axis=1)
            sorted_scores_all.append(sorted_scores[:,:,np.newaxis])
        # Sum the scores of each chunk on each retriever
        summed_scores = np.sum(np.concat(sorted_scores_all, axis=2), axis=2)
        # Sort the chunks indexes and scores by their summed scores
        sorted_indices = np.argsort(-summed_scores, axis=1) # - sign for descending order
        sorted_scores = np.take_along_axis(summed_scores, sorted_indices, axis=1)

        return sorted_indices.tolist(), sorted_scores.tolist()

    @staticmethod
    def rescale_scores_with_rrf(ranked_scores: list[float], k_value: int = 60) -> list[float]:
        """Uses the reciprocal rank formula to rescale the scores.

        Args:
            ranked_scores (list[float]): the scores to rerank (values actually doesn't matter, as only the rank is used.)
            k_value (int): the value of k in the formula.

        Returns:
            list[float]: the new scores
        """
        return [1/(k_value + i) for i in range(len(ranked_scores))]


    @staticmethod
    def rescale_scores_with_min_max(scores: list[float]) -> list[float]:
        """Normalizes the scores of the chunks returned by a retriever
        between 0 an 1.

        Args:
            scores (list[float]): the list of scores to normalize

        Returns:
            list[float]: the normalized scores
        """
        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            return [1 for _ in scores] # avoid division by zero
        return [(score - min_score) / (max_score - min_score) for score in scores]
