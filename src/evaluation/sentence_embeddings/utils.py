import numpy.typing as npt
import numpy as np

from src.components import Chunk, ReferenceSection
from src.evaluation.sentence_embeddings.model import model


def calculate_norm(embedding_matrix: npt.NDArray[np.float32]):
    """
    Calculate the norm of the given embedding matrix.

    Args:
        embedding_matrix (npt.NDArray[np.float64]): The embedding matrix for which to calculate the norm.

    Returns:
        np.float32: The norm of the embedding matrix. If the input is a 1D array, returns a single norm value.
                       If the input is a 2D array, returns the norm for each row, keeping the dimensions.
    """
    if embedding_matrix.ndim == 1:
        return np.linalg.norm(embedding_matrix)
    return np.linalg.norm(embedding_matrix, axis=1, keepdims=True)


def calculate_cosine_similarity(
    embedding_target_passages: npt.NDArray[np.float32],
    embedding_comparison: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Calculate the cosine similarity between two sets of embeddings.

    Args:
        embedding_target_passages (npt.NDArray[np.float32]): The target embeddings.
        embedding_comparison (npt.NDArray[np.float32]): The comparison embeddings.

    Returns:
        float: The cosine similarity score between the target and comparison embeddings.
    """
    normalized_embedding_target_passages = embedding_target_passages / calculate_norm(
        embedding_target_passages
    )
    normalized_embedding_comparison = embedding_comparison / calculate_norm(
        embedding_comparison
    )
    return np.dot(
        normalized_embedding_target_passages, normalized_embedding_comparison.T
    )


def retrieve_top_k_chunks(
    list_references: list[ReferenceSection],
    list_chunks: list[Chunk],
    list_embeddings: npt.NDArray[np.float32],
    top_k: int,
) -> list[list[Chunk]]:
    """
    Retrieve the top-k chunks that are most similar to the given reference passages.

    Args:
        list_references (list[ReferenceSection]): A list of reference sections, each containing a target passage.
        list_chunks (list[Chunk]): A list of chunks, each containing text to be compared.
        list_embeddings (npt.NDArray[np.float32]): A numpy array containing the embeddings for the chunks.
        top_k (int): The number of top chunks to retrieve for each reference passage.

    Returns:
        list[list[Chunk]]: A list of lists, where each inner list contains the top-k chunks that are most similar to the reference passages.
    """
    list_references_query = np.array([reference.query for reference in list_references])

    embeddings_chunks = list_embeddings
    embeddings_reference = model.get_embeddings(list_references_query)

    cosim_values = calculate_cosine_similarity(
        embeddings_chunks, embeddings_reference
    ).T

    top_k_indices = np.argsort(cosim_values, axis=1)[:, -top_k:][:, ::-1]
    return [[list_chunks[idx] for idx in row] for row in top_k_indices]
