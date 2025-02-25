import json
import re
import datasets
import numpy as np
from sklearn.metrics import ndcg_score
from torch import topk
from tqdm import tqdm
from unidecode import unidecode
from sentence_transformers.util import cos_sim

def chunks_to_dataset(
    path_to_chunks_jsonfile : str
    ) -> datasets.DatasetDict:
    """Builds a DatasetDict object from chunks saved in json file obtained from the an evaluator.get_chunks()

    Args:
        path_to_chunks_jsonfile (str): the filepath to the json file where chunks are stored.

    Returns:
        datasets.DatasetDict: a dataset dict where each combination of parser/chunker as a split.
    """
    dataset_dict = datasets.DatasetDict()
    with open(path_to_chunks_jsonfile, encoding="utf8") as file:
        all_chunks = json.load(file)
    for pipeline_name in all_chunks.keys():
        for chunker_name, chunks in all_chunks[pipeline_name].items():
            dataset = datasets.Dataset.from_list(chunks)
            split_name = pipeline_name + "__" + chunker_name
            dataset_dict[split_name] = dataset

    return dataset_dict

def retrieve_chunks(query_sample: datasets.Dataset, chunks_dataset: datasets.Dataset, k : int =10) -> datasets.Dataset:
    """Considering a sample of the "queries" dataset, retrieves k chunks.
    Both query_sample and "chunks_dataset" must have a "emb" column which is the embedding.

    Returns:
        dataset.Dataset: a dataset where each sample a chunk retrieved.
    """
    sim_matrix = cos_sim(query_sample["emb"], chunks_dataset["emb"])
    topk_sim_scores, topk_indices = topk(sim_matrix, k, dim=1)

    top_chunks = chunks_dataset.select(topk_indices.squeeze().tolist())
    top_chunks = top_chunks.add_column("cosim", topk_sim_scores.squeeze().tolist())

    return top_chunks

def _map_labeled_passage_to_chunk(queries_dataset: datasets.Dataset, chunks_dataset: datasets.Dataset, rouge_threshold: float = .7) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Adds a column "chunks_idx" to the queries dataset
    that contains the corresponding indexes of the chunks
    that contain the passages labeled as relevant.

    Args:
        queries_dataset (datasets.Dataset): the dataset of queries.
        chunks_dataset (datasets.Dataset): the dataset of chunks.
        rouge_threshold (float, optional): the minimum score rouge score between the chunk's text
            and labeled passage to consider the chunk contains the passage. Defaults to .7.

    Returns:
        tuple(datasets.Dataset, datasets.Dataset): the queries dataset with the "chunks_idx" column added
            and the chunks dataset with a "idx" column added.
    """
    # create masks from chunks features
    filenames_chunks = np.array(chunks_dataset["source_file"])
    page_start_chunks = np.array(chunks_dataset["page_start"])
    page_end_chunks = np.array(chunks_dataset["page_end"])
    # store results in buffer
    column_buffer = []
    for query_sample in queries_dataset:
        # Get a list of tuples (source_doc, page, passage)
        passage_filename_page_combinations = [
            (filename, page, passage)
            for filename, target_pages, target_passages in zip(
                query_sample["source_file"],
                query_sample["target_pages"],
                query_sample["target_passages"],
                )
            for page, passage in zip(target_pages, target_passages)
        ]
        # creates masks from list of tuples
        filename_mask, page_mask, passages = zip(*passage_filename_page_combinations)
        filename_mask, page_mask = np.array(filename_mask)[:, np.newaxis], np.array(page_mask)[:, np.newaxis]
        # find pairs of potential passage-chunk matches
        passages_idx, chunks_idx = np.where(
            (filename_mask == filenames_chunks) & 
            (page_start_chunks <= page_mask) & 
            (page_end_chunks >= page_mask)
        )
        # get the list of chunks labeled as relevant for the query
        chunks_idx_of_queries = [
            chunk_idx for chunk_idx, passage_idx in zip(chunks_idx, passages_idx)
            if get_rouge_score(passages[int(passage_idx)], chunks_dataset[int(chunk_idx)]["text"]) >= rouge_threshold
        ]
        column_buffer.append(chunks_idx_of_queries)
    
    queries_dataset = queries_dataset.add_column("chunks_idx", column_buffer)
    chunks_dataset = chunks_dataset.add_column("idx", list(range(len(chunks_dataset))))

    return queries_dataset, chunks_dataset


def get_rouge_score(passage_text: str, chunk_text: str) -> float:
    """Computes ROUGE score on unigrams

    Args:
        passage_text (str): the target passage
        chunk_text (str): the text of the chunk

    Returns:
        float: A score. 1 if all word of passage are in chunk
    """
    norm_passage = unidecode(passage_text.lower())
    norm_chunk = unidecode(chunk_text.lower())
    passage_words = re.findall(r"\w+", norm_passage)

    return len([word for word in passage_words if word in norm_chunk]) / len(passage_words)


def _compute_recall(OK_chunks_idxes: list[int], top_chunks_indexes: list[int], k : int = 10) -> float:
    """Considering a query, computes the recall.

    Args:
        OK_chunks_idxes (list[int]): the indexes of the chunks labeled as relevant.
        top_chunks_indexes (list[int]):  the indexes of the chunks retrieved sorted by similarity scores. (return of torch.topk)
        k (int, optional): mount of chunks to consider to compute recall. Defaults to 10.

    Returns:
        float: the recall.
    """
    return (
        len([idx for idx in top_chunks_indexes[:k] if idx in OK_chunks_idxes])
        / len(set(OK_chunks_idxes))
        ) if OK_chunks_idxes else 0


def _compute_ndcg(OK_chunks_idxes: list[int], top_chunks_indexes: list[int], top_cosims_values: list[float], k : int = 10) -> float:
    """Considering a query, computes the NDCG.

    Args:
        OK_chunks_idxes (list[int]): the indexes of the chunks labeled as relevant.
        top_chunks_indexes (list[int]): the indexes of the chunks retrieved sorted by similarity scores. (return of torch.topk)
        top_cosims_values (list[float]): the sorted similarity scores. (return of torch.topk)
        k (int, optional): amount of chunks to consider to compute NDCG. Defaults to 10.

    Returns:
        float: the NDCG score.
    """
    # Get array of True/False if chunk is labaled relevant
    correct_chunks = [idx in OK_chunks_idxes for idx in top_chunks_indexes]
    return ndcg_score(
            [correct_chunks],
            [top_cosims_values],
            k=k)


def run_scoring(queries_dataset: datasets.Dataset, chunks_dataset: datasets.Dataset, k:int=10) -> tuple[float,float]:
    """Returns metrics considering a dataset of queries and chunks.

    Args:
        queries (datasets.Dataset): the dataset of queries.
        chunks (datasets.Dataset): the dataset of chunks.

    Returns:
        tuple[float,float]: recall and ndcg
    """
    datasets.disable_progress_bars()
    sim_matrix = cos_sim(queries_dataset["emb"], chunks_dataset["emb"])
    topk_sim_scores, topk_indices = topk(sim_matrix, len(chunks_dataset), dim=1)
    queries_dataset, chunks_dataset = _map_labeled_passage_to_chunk(
        queries_dataset, chunks_dataset
        )
    metrics: list[tuple[float, float]] = [(
        _compute_recall(
            query_sample["chunks_idx"],
            topk_indices[i].tolist(),
            k
            ),
        _compute_ndcg(
            query_sample["chunks_idx"],
            topk_indices[i].tolist(),
            topk_sim_scores[i].tolist(),
            k
            )
        )
        for i, query_sample in enumerate(tqdm(queries_dataset))
    ]
    recalls, ndcgs = zip(*metrics)

    return float(np.mean(recalls)), float(np.mean(ndcgs))