import os
import numpy.typing as npt
import numpy as np

from dotenv import load_dotenv, find_dotenv

# from model2vec import StaticModel
from chromacache import ChromaCache
from chromacache.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv(find_dotenv())


class EmbeddingModel:
    def __init__(self, embedding_model: str):
        self.model = None
        self.embedding_name = embedding_model
        self._get_model()

    def _get_model(self):
        """
        Initialize and return the appropriate model based on the embedding type.

        This method sets the `self.model` attribute to either a static embedding model or a dynamic
        embedding model, depending on the value of `self.static_embedding`.

        Returns:
            None
        """
        if self.embedding_name == "bge-base-en-v1.5":
            self.model = ChromaCache(
                SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
            )
        elif self.embedding_name == "snowflake-arctic-embed-m-v2.0":
            self.model = ChromaCache(
                SentenceTransformerEmbeddingFunction(
                    model_name="Snowflake/snowflake-arctic-embed-m-v2.0",
                    trust_remote_code=True,
                )
            )
        else:
            raise ValueError(f"Model {self.embedding_name} not implemented!")

    def get_embeddings(self, sentences: str | list[str]) -> npt.NDArray[np.float32]:
        """
        Generate embeddings for the given sentences using a pre-trained model.

        Args:
            sentences (str | list[str]): A single sentence or a list of sentences to encode.

        Returns:
            NDArray[np.float32]: The embeddings for the input sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.model.encode(sentences)
        if isinstance(embeddings, list):
            return np.array(embeddings)
        return embeddings


model = EmbeddingModel(embedding_model=os.getenv("EMBEDDING"))
