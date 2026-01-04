from typing import Any, Callable
import datasets
from datasets.features.features import Value
from chromacache import ChromaCache
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch import topk # pylint: disable=no-name-in-module

from src.retrievers.abs_retriever import AbsRetriever

class DenseRetriever(AbsRetriever):
    """A basic dense retriever. It uses embedding and cosine similarity to retrieve most relevant chunks"""
    def __init__(
            self,
            queries_embedding_function: Callable[[list[str]], list[list[float]]],
            documents_embedding_function : Callable[[list[str]], list[list[float]]],
            default_k : int = 10,
            queries_dataset: datasets.Dataset | None = None,
            chunks_dataset: datasets.Dataset | None = None,
            description :str = ""
            ):
        """Initialises a dense retriever. Dense retriever uses embedding

        Args:
            queries_embedding_function: (Callable): The function to use to get the embeddings of queries.
                NOTE: Must takes a list[str] as input and returns a list of embeddings.
            documents_embedding_function: (Callable): The function to use to get the embeddings of documents.
                NOTE: Must takes a list[str] as input and returns a list of embeddings.
            default_k (int): the amount of chunks to retrieve by default when calling to retriever. Defaults to 10.
            queries_dataset (datasets.Dataset | None): the dataset of queries. Can be set to None if you plan to provided them later.
                NOTE: Must have a column "query" of type str which contains the queries.
            chunks_dataset (datasets.Dataset | None): the dataset which contains the chunks. Can be set to None if you plan to provide them later.
                NOTE: Must have a column "text" of type str which contains the text of the chunks.
            description (str |None): an additional description of the retriever. Mainly used to
                keep tracks of which retriever is which during evaluation. Defaults to None.
        """
        self.queries_embedding_function = queries_embedding_function
        self.documents_embedding_function = documents_embedding_function
        super().__init__(default_k=default_k, queries_dataset=queries_dataset, chunks_dataset=chunks_dataset, description=description)

    @property
    def queries_dataset(self):
        """The dataset of queries"""
        if self._queries_dataset is None:
            raise ValueError("The queries_dataset must be set before accessing it.")
        return self._queries_dataset

    @queries_dataset.setter
    def queries_dataset(self, dataset: datasets.Dataset):
        if dataset is not None:
            DenseRetriever._validate_dataset_schema(dataset, {"query": Value("string")})
            dataset = dataset.add_column("emb", self.queries_embedding_function(dataset["query"]))
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
            DenseRetriever._validate_dataset_schema(dataset, {"text": Value("string")})
            dataset = dataset.add_column("emb", self.documents_embedding_function(dataset["text"]))
        self._chunks_dataset = dataset


    def retrieve_chunks(
        self, query: str, k: int | None = None
    ) -> datasets.Dataset:
        """Considering a query, retrieves the k most relevant chunks.
        Both query_sample and "chunks_dataset" must have a "emb" column which is the embedding.

        Args:
            query (str): the query to retrieve chunks for, as a string.
            k (int): the amount of chunks to retrieve. If left as None, uses the default_k. Defaults to None.

        Returns:
            dataset.Dataset: a dataset where each sample a chunk retrieved.
        """
        k = k or self.default_k
        query_emb = self.queries_embedding_function([query])
        sim_matrix = cos_sim(
            np.array(query_emb, dtype=np.float32),
            np.array(self.chunks_dataset["emb"], dtype=np.float32)
            )
        topk_sim_scores, topk_indices = topk(sim_matrix, min(k, len(self.chunks_dataset)), dim=1)

        top_chunks = self.chunks_dataset.select(topk_indices.squeeze().tolist())
        top_chunks = top_chunks.add_column("score", topk_sim_scores.squeeze().tolist())

        return top_chunks


    def rank_chunks_by_relevance_2d(self) -> tuple[list[list[int]], list[list[float]]]:
        """Considering the queries and chunks datasets,
        ranks the chunks by relevance regarding the queries.
        It returns results as 2D matrices of shape (len(queries), len(chunks)):
        - The first matrix corresponds to the indices of the chunks ranked by relevance regarding the query. 
        - The second matrix correspond to the relevance scores.

        Returns:
            tuple[list[list[int]], list[list[float]]]: respecively the indices of the chunks
                sorted by relevance regarding each query,
                and the corresponding scores.
        """
        cosims_matrix = cos_sim(self.queries_dataset["emb"], self.chunks_dataset["emb"])
        topk_sim_scores, topk_indices = topk(cosims_matrix, len(self.chunks_dataset), dim=1)

        return topk_indices.squeeze().tolist(), topk_sim_scores.squeeze().tolist()
