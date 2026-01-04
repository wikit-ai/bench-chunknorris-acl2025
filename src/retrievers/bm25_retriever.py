import datasets
import Stemmer
import bm25s
from datasets.features.features import Value
from src.retrievers.abs_retriever import AbsRetriever

class BM25Retriever(AbsRetriever):
    stemmer: Stemmer

    def __init__(
            self,
            queries_dataset: datasets.Dataset | None = None,
            chunks_dataset: datasets.Dataset | None = None,
            description :str = ""
            ):
        """Initialises a bm25 retriever. BM25 retriever uses BM25 algorithm to retrieve most relevant chunks.

        Args:
            queries_dataset (datasets.Dataset | None): the dataset of queries. Can be set to None if you plan to provided them later.
                NOTE: Must have a column "query" of type str which contains the queries.
            chunks_dataset (datasets.Dataset | None): the dataset which contains the chunks. Can be set to None if you plan to provide them later.
                NOTE: Must have a column "text" of type str which contains the text of the chunks.
            description (str |None): an additional description of the retriever. Mainly used to
                keep tracks of which retriever is which during evaluation. Defaults to None.
        """
        self.stemmer = Stemmer.Stemmer("french")
        self._bm25 = bm25s.BM25()
        super().__init__(queries_dataset=queries_dataset, chunks_dataset=chunks_dataset, description=description)

    @property
    def chunks_dataset(self):
        """The dataset of chunks"""
        if self._chunks_dataset is None:
            raise ValueError("The chunks_dataset must be set before accessing it.")
        return self._chunks_dataset

    @chunks_dataset.setter
    def chunks_dataset(self, dataset: datasets.Dataset):
        if dataset is not None:
            BM25Retriever._validate_dataset_schema(dataset, {"text": Value("string")})
            corpus_tokens = bm25s.tokenize(dataset["text"], stopwords="fr", stemmer=self.stemmer)
            self._bm25.index(corpus_tokens)
        self._chunks_dataset = dataset


    def retrieve_chunks(
        self, query: str, k: int = 10
    ) -> datasets.Dataset:
        """Considering a query, retrieves the k most relevant chunks.

        Args:
            query (str): the query to retrieve chunks for, as a string.
            k (int): the amount of chunks to retrieve. Defaults to 10.

        Returns:
            dataset.Dataset: a dataset where each sample a chunk retrieved.
        """
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        topk_indexes, topk_scores = self._bm25.retrieve(query_tokens, k=min(k, len(self.chunks_dataset)))

        top_chunks = self.chunks_dataset.select(topk_indexes.squeeze().tolist())
        top_chunks = top_chunks.add_column("score", topk_scores.squeeze().tolist())

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
        query_tokens = bm25s.tokenize(self.queries_dataset["query"], stemmer=self.stemmer)
        topk_indexes, topk_scores = self._bm25.retrieve(query_tokens, k=len(self.chunks_dataset))

        return topk_indexes.squeeze().tolist(), topk_scores.squeeze().tolist()