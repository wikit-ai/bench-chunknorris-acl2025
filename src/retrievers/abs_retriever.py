from abc import ABC, abstractmethod

import datasets
from datasets.features.features import Value, FeatureType


class AbsRetriever(ABC):
    """Abstract class of a retriever."""
    def __init__(
            self,
            default_k : int = 10,
            queries_dataset: datasets.Dataset | None = None,
            chunks_dataset: datasets.Dataset | None = None,
            description: str = ""
            ):
        """Initialises a dense retriever. Dense retriever uses embedding
            default_k (int): the amount of chunks to retrieve by default when calling to retriever. Defaults to 10.
            queries_dataset (datasets.Dataset): the dataset of queries.
                NOTE: Must have a column "query" of type str which contains the queries.
            chunks_dataset (datasets.Dataset): the dataset which contains the chunks.
                NOTE: Must have a column "text" of type str which contains the text of the chunks.
            description (str |None): an additional description of the retriever. Mainly used to
                keep tracks of which retriever is which during evaluation. Defaults to None.
        """
        self.default_k = default_k
        self._queries_dataset = None
        self.queries_dataset = queries_dataset # Go through setter
        self._chunks_dataset = None
        self.chunks_dataset = chunks_dataset # Go through setter
        self.description = description

    @property
    def queries_dataset(self):
        """The dataset of queries"""
        if self._queries_dataset is None:
            raise ValueError("The queries_dataset must be set before accessing it.")
        return self._queries_dataset
    
    @queries_dataset.setter
    def queries_dataset(self, dataset: datasets.Dataset):
        if dataset is not None:
            AbsRetriever._validate_dataset_schema(dataset, {"query": Value("string")})
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
            AbsRetriever._validate_dataset_schema(dataset, {"text": Value("string")})
        self._chunks_dataset = dataset

    @staticmethod
    def _validate_dataset_schema(dataset: datasets.Dataset, columns_mapping: dict[str, FeatureType]):
        """Used to validate that the dataset as expected schema.

        Args:
            dataset (datasets.Dataset): the dataset to validate
            columns_mapping (dict[str, FeatureType]): the mapping of column name and type to validate.

        Raises:
            ValueError: If column not present in dataset.
            ValueError: If column has wrong type.
        """
        for col_name, col_type in columns_mapping.items():
            if not col_name in dataset.features.keys():
                raise ValueError(f"Column '{col_name}' missing in dataset.")
            if not col_type == dataset.features[col_name]:
                raise ValueError(f"Column '{col_name}' should have type '{col_type}'. Got '{dataset.features[col_name]}' instead.")
            
    @abstractmethod
    def retrieve_chunks(
        self, query: str, k: int | None = None
    ) -> datasets.Dataset:
        """Considering a query, retrieves the k most relevant chunks.

        Args:
            query (str): the query to retrieve chunks for, as a string.
            k (int | None): the amount of chunks to retrieve. If left as None, uses the default_k. Defaults to None.

        Returns:
            dataset.Dataset: a dataset where each sample a chunk retrieved.
        """
        k = k or self.default_k


    @abstractmethod
    def rank_chunks_by_relevance_2d(self) -> tuple[list[list[int]], list[list[float]]]:
        """Considering the queries and chunks datasets,
        ranks the chunks by relevance regarding the queries.
        It returns results 2D matrices of shape (len(queries), len(chunks)):
        - The first matrix corresponds to the indices of the chunks ranked by relevance regarding the query. 
        - The second matrix correspond to the relevance scores.

        Returns:
            tuple[list[list[int]], list[list[float]]]: respecively the indices of the chunks
                sorted by relevance regarding each query,
                and the corresponding scores.
        """
