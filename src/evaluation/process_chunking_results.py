import json
import re
import datasets
import numpy as np
from sklearn.metrics import ndcg_score
from torch import topk
from tqdm import tqdm
from unidecode import unidecode
from sentence_transformers.util import cos_sim


def retrieve_chunks(query_sample, chunks_dataset, k : int =10) -> list[datasets.Dataset]:
    """Both dataset must have a "emb" column which is the embedding.
    
    Returns:
        list[dataset.Dataset]: a list of datasets 
            where each dataset is the subset of k chunks retrieved
    """
    sim_matrix = cos_sim(query_sample["emb"], chunks_dataset["emb"])
    topk_sim_scores, topk_indices = topk(sim_matrix, k, dim=1)
    
    return datasets.Dataset.from_dict(
        chunks_dataset[topk_indices.squeeze()] | {"cosim" : topk_sim_scores.squeeze()}
        )

def flag_retrieved_chunks(query_sample: datasets.Dataset, retrieved_chunks: list[datasets.Dataset], rouge_threshold: float = 0.7):
    """For each query and retrieved chunks, flags the chunks that were labelled relevant.
    
    Args: 
        query_sample: a sample (row) of the queries dataset
        retrieved_chunks: list of dataset. Each dataset contains topk retrieved chunks for each query
    """
    # Store future columns
    is_correct_buffer = [False]* len(retrieved_chunks) # if chunk is correct
    matches_passage_idx_buffer = [[] for _ in range(len(retrieved_chunks))] # which chunk passage matches
    matches_passage_text_buffer = [[] for _ in range(len(retrieved_chunks))] # which chunk passage matches

    # get a list of tuple (source file, target page, target passage) from labeled dataset
    # do this to unnnest the list of list.
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
    filename_mask, page_mask, passages = list(zip(*passage_filename_page_combinations)) 
    filename_mask, page_mask = np.array(filename_mask)[:, np.newaxis], np.array(page_mask)[:, np.newaxis]

    filenames_chunks = np.array(retrieved_chunks["source_file"])
    page_start_chunks = np.array(retrieved_chunks["page_start"])
    page_end_chunks = np.array(retrieved_chunks["page_end"])

    # Get chunks that might correspond to a labeled page
    # Note : page might be ok if it is from labeled document and contains the labeled page.
    passage_idxes, chunks_idx = np.where(
        (filename_mask == filenames_chunks) & 
        (page_start_chunks <= page_mask) & 
        (page_end_chunks >= page_mask)
    )

    # For each of the chunks that might be correct, check if it contain the target passage
    for passage_idx, chunk_idx in zip(passage_idxes, chunks_idx):
        passage_text = passages[passage_idx]
        chunk_text = retrieved_chunks["text"][chunk_idx]
        score = get_rouge_score(passage_text, chunk_text)
        if score >= rouge_threshold:
            is_correct_buffer[chunk_idx] = True
            matches_passage_idx_buffer[chunk_idx].append(passage_idx)

    # Store results in new columns
    retrieved_chunks = retrieved_chunks.add_column("is_correct", is_correct_buffer)
    retrieved_chunks = retrieved_chunks.add_column("matches_passage_idx", matches_passage_idx_buffer)
    retrieved_chunks = retrieved_chunks.add_column("matches_passage_text", matches_passage_text_buffer)

    return retrieved_chunks


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


def compute_recall(query_sample: datasets.Dataset, retrieved_chunks: list[datasets.Dataset], k=10):
    """Provided the flagged retrieved chunks and the queries dataset,
    computes the recall"""
    return (
            len({idx for chunk in retrieved_chunks.select(range(k)) for idx in chunk["matches_passage_idx"]}) # retrieved passages
            / len([passage for passages in query_sample["target_passages"] for passage in passages]) # total passages
        )


def compute_ndcg(retrieved_chunks: list[datasets.Dataset], k : int = 10):
    """From NDCG
    """
    return ndcg_score(
            np.array([retrieved_chunks["is_correct"]]),
            np.array([retrieved_chunks["cosim"]]),
            k=k)


def export_res(queries_dataset, retrieved_chunks):
    output = []
    queries_dataset = queries_dataset.remove_columns(["emb"])
    for queries, chunks in zip (queries_dataset, retrieved_chunks):
        chunks = chunks.remove_columns(["emb"])
        output.append(queries | {"retrieved_chunks" : chunks.to_list()})

    with open("retrieval_results.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)


def run_scoring(queries : datasets.Dataset, chunks_dataset: datasets.Dataset) -> tuple[float, float]:
    """Returns metrics considering a dataset of queries and chunks

    Args:
        queries (datasets.Dataset): 
        chunks (datasets.Dataset): _description_

    Returns:
        tuple[float, float]: recall and ndcg
    """
    sim_matrix = cos_sim(queries["emb"], chunks_dataset["emb"])
    topk_sim_scores, topk_indices = topk(sim_matrix, len(chunks_dataset), dim=1)
    recalls, ndcgs = [], []
    for i, query_sample in enumerate(tqdm(queries)):
        top_chunks = datasets.Dataset.from_dict(
        chunks_dataset[topk_indices[i]] | {"cosim" : topk_sim_scores[i]}
        )
        top_chunks = flag_retrieved_chunks(query_sample, top_chunks)
        recalls.append(compute_recall(query_sample, top_chunks, 10))
        ndcgs.append(compute_ndcg(top_chunks, 10))

    return float(np.mean(recalls)), float(np.mean(ndcgs))