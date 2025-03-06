from typing import Any
from collections import defaultdict
from datetime import datetime
import json
import os
import re

import datasets
import numpy as np

from huggingface_hub import HfApi
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from unidecode import unidecode


from src.pipelines.abs_pipeline import AbsPipeline
from src.chunkers.abs_chunker import AbstractChunker
from src.retrievers.abs_retriever import AbsRetriever
from src.components import Chunk
from src.utils import LOGGER


class ChunkingEvaluator:
    """Meant to run an evaluation on a parser and set of chunkers."""

    timestamp: str

    def __init__(
        self,
        pipeline: AbsPipeline,
        chunkers: list[AbstractChunker | None],
        retrievers: list[AbsRetriever],
        results_dir: str = "./results",
    ):
        """Instanciate an evaluator.

        Args:
            pipeline (AbsPipeline): the pipeline to be tested. Must inherit from AbsPipeline
            chunkers (list[AbstractChunker] | None): list of chunker to be tester.
                NOTE: If "None" is passed in the list, then the default_chunker of the pipeline will be used.
            retrievers (list[AbsRetriever]) : a list of retrievers to use for evaluation.
            sentence_transformer_hf_repo (str): the HF repo of a model compatible with SentenceTransformer.
                Used to embed chunks and queries to compute metrics. Defaults to BAAI/bge-small-en-v1.5.
        """
        self.pipeline = pipeline
        self.chunkers = chunkers
        self.retrievers = retrievers
        self.results_dir = self._set_result_dir(results_dir)

    def _set_result_dir(self, results_dir) -> str:
        """Set the directory in which results will be stored"""
        self.timestamp = str(datetime.now()).replace(":", "-").replace(".", "-")
        results_dir = os.path.join(results_dir, "chunking", self.timestamp)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        LOGGER.info("Results will be stored in %s", results_dir)

        return results_dir

    def save_as_json(self, dict_to_save: dict[Any, Any], filename: str):
        """Saves a dict as a JSON file"""
        if not filename.endswith(".json"):
            raise ValueError("Provided filename must end with .json")
        with open(
            os.path.join(self.results_dir, filename), "w", encoding="utf8"
        ) as file:
            json.dump(dict_to_save, file, indent=4, ensure_ascii=False)

    def evaluate_chunking(
        self,
        queries_dataset: datasets.Dataset,
        pdf_filepaths: list[str] | None = None,
        path_to_chunks: str | None = None,
    ):
        """Runs an experiment.

        Args:
            queries_dataset (datasets.Dataset | None): the dataset a queries to use for evaluation.
                NOTE: The dataset schema (column names, types, and overall structure) must be equivalent
                to Wikit's PIRE dataset. For more info see : https://huggingface.co/datasets/Wikit/PIRE.
                If None, Wikit/PIRE dataset will be used, assuming the pdf filepaths point to the PDF files from this dataset.
            pdf_filepaths (list[str] | None): the list of filepaths pointing to pdf files of the evaluation dataset. If None,
                you must provided "path_to_chunks" to reuse the chunks obtained from previous runs.
            path_to_chunks (str | None): the path to the json file where chunks from previous runs are stored.
        """
        if pdf_filepaths is not None:
            _ = self.get_chunks(pdf_filepaths)
            chunks_datadict = ChunkingEvaluator.chunks_to_dataset(
                os.path.join(self.results_dir, "chunks.json")
            )
        elif path_to_chunks is not None:
            chunks_datadict = ChunkingEvaluator.chunks_to_dataset(
                path_to_chunks
            )
        else:
            raise ValueError(
                "Either pdf_filepaths or path_to_chunks must be provided."
            )
        self.run_chunking_evaluation(chunks_datadict, queries_dataset)

    def get_chunks(
        self, pdf_filepaths: list[str]
    ) -> dict[str, list[Chunk]]:
        """Considering the pipeline and the list of chunkers provided to the evaluator,
        gets the chunks obtained for each files and for each chunker.

        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing the pdfs of the retrieval dataset.
                They can be found here: https://huggingface.co/datasets/Wikit/PIRE/tree/main

        Returns:
            dict[str,list[Chunk]] : a dict with the chunker's name as key
                and the list chunks of all documents as value.
        """
        chunks_dict: dict[str, list[Chunk]] = defaultdict(list)
        # parse the file
        for filepath in tqdm(pdf_filepaths):
            self.pipeline.parse_file(filepath)
            # use the result of the parsing to chunk with the chunkers
            for chunker in self.chunkers:
                self.pipeline.external_chunker = chunker
                chunker_name = (
                    "Default"
                    if chunker is None
                    else self.pipeline.external_chunker.__class__.__name__
                )
                chunks = self.pipeline.chunk()
                chunks_dict[chunker_name].extend(c for c in chunks if c.text)

        pipeline_name = self.pipeline.__class__.__name__
        dumped_chunks = {
            pipeline_name: {
                chunker: [chunk.model_dump() for chunk in chunks]
                for chunker, chunks in chunks_dict.items()
            }
        }
        self.save_as_json(dumped_chunks, "chunks.json")

        return chunks_dict


    @staticmethod
    def chunks_to_dataset(path_to_chunks_jsonfile: str) -> datasets.DatasetDict:
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


    def run_chunking_evaluation(
        self,
        chunks_datadict: datasets.DatasetDict,
        queries_dataset: datasets.Dataset,
    ):
        """Runs an evaluation to assess the chunking performance.

        Args:
            chunks_datadict (datasets.DatasetDict): a DatasetDict of the chunks, where:
                - each split has name as: parsername__chunkername.
                - a column name "text" contains the chunks' text.
            queries_dataset (datasets.Dataset): the dataset a queries to use for evaluation.
                NOTE: The dataset schema (column names, types, and overall structure) must be equivalent
                to Wikit's PIRE dataset. For more info see : https://huggingface.co/datasets/Wikit/PIRE
        """
        results: list[dict[str, str | float]] = []
        for retriever in self.retrievers:
            retriever.queries_dataset = queries_dataset
            for split in chunks_datadict.keys():
                parser_name, chunker_name = split.split("__")
                chunks = chunks_datadict[split]
                retriever.chunks_dataset = chunks

                recalls, ndcgs = ChunkingEvaluator.run_scoring(
                    queries_dataset, chunks, retriever, retriever.default_k
                    )

                results.append(
                    {
                        "parser": parser_name,
                        "chunker": chunker_name,
                        "retriever": retriever.__class__.__name__,
                        "retriever_description": retriever.description,
                        "dataset_name": queries_dataset.info.dataset_name,
                        "dataset_subset": queries_dataset.config_name,
                        "dataset_split": str(queries_dataset.split),
                        f"recalls@{retriever.default_k}": recalls,
                        f"recall_mean@{retriever.default_k}": float(np.mean(recalls)),
                        f"ndcgs@{retriever.default_k}": ndcgs,
                        f"ndcg_mean@{retriever.default_k}": float(np.mean(ndcgs)),
                    }
                )

        self.save_as_json(results, f"{str(queries_dataset.split)}_results.json")


    def push_results_to_hf(self, hf_repo_id: str):
        """Pushes the results to huggingface"""
        api = HfApi()
        for filepath in os.listdir(self.results_dir):
            api.upload_file(
                path_or_fileobj=os.path.join(self.results_dir, filepath),
                path_in_repo=os.path.join("chunking", self.timestamp, filepath),
                repo_id=hf_repo_id,
                repo_type="dataset",
            )


    @staticmethod
    def _map_labeled_passage_to_chunk(
        queries_dataset: datasets.Dataset,
        chunks_dataset: datasets.Dataset,
        rouge_threshold: float = 0.7,
    ) -> tuple[datasets.Dataset, datasets.Dataset]:
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
            filename_mask, page_mask = (
                np.array(filename_mask)[:, np.newaxis],
                np.array(page_mask)[:, np.newaxis],
            )
            # find pairs of potential passage-chunk matches
            passages_idx, chunks_idx = np.where(
                (filename_mask == filenames_chunks)
                & (page_start_chunks <= page_mask)
                & (page_end_chunks >= page_mask)
            )
            # get the list of chunks labeled as relevant for the query
            chunks_idx_of_queries = [
                chunk_idx
                for chunk_idx, passage_idx in zip(chunks_idx, passages_idx)
                if ChunkingEvaluator.get_rouge_score(
                    passages[int(passage_idx)], chunks_dataset[int(chunk_idx)]["text"]
                )
                >= rouge_threshold
            ]
            column_buffer.append(chunks_idx_of_queries)

        queries_dataset = queries_dataset.add_column("chunks_idx", column_buffer)
        chunks_dataset = chunks_dataset.add_column("idx", list(range(len(chunks_dataset))))

        return queries_dataset, chunks_dataset


    @staticmethod
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

        return len([word for word in passage_words if word in norm_chunk]) / len(
            passage_words
        )


    @staticmethod
    def _compute_recall(
        OK_chunks_idxes: list[int], top_chunks_indexes: list[int], k: int = 10
    ) -> float:
        """Considering a query, computes the recall.

        Args:
            OK_chunks_idxes (list[int]): the indexes of the chunks labeled as relevant.
            top_chunks_indexes (list[int]):  the indexes of the chunks retrieved sorted by similarity scores. (return of torch.topk)
            k (int, optional): mount of chunks to consider to compute recall. Defaults to 10.

        Returns:
            float: the recall.
        """
        return (
            (
                len([idx for idx in top_chunks_indexes[:k] if idx in OK_chunks_idxes])
                / len(set(OK_chunks_idxes))
            )
            if OK_chunks_idxes
            else 0
        )

    @staticmethod
    def _compute_ndcg(
        OK_chunks_idxes: list[int],
        top_chunks_indexes: list[int],
        top_cosims_values: list[float],
        k: int = 10,
    ) -> float:
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
        return ndcg_score([correct_chunks], [top_cosims_values], k=k)

    @staticmethod
    def run_scoring(
        queries_dataset: datasets.Dataset, chunks_dataset: datasets.Dataset, retriever: AbsRetriever, k: int = 10
    ) -> tuple[list[float],list[float]]:
        """Returns metrics considering a dataset of queries and chunks.

        Args:
            queries (datasets.Dataset): the dataset of queries.
            chunks (datasets.Dataset): the dataset of chunks.
            retriever (AbsRetriever): the retriever to use.

        Returns:
            tuple[list[float],list[float]]: the recalls and ndcgs for each query in dataset.
        """
        datasets.disable_progress_bars()
        queries_dataset, chunks_dataset = ChunkingEvaluator._map_labeled_passage_to_chunk(
            queries_dataset, chunks_dataset
        )
        ranked_chunks_idxes, ranked_chunks_scores = retriever.rank_chunks_by_relevance_2d()
        metrics: list[tuple[float, float]] = [
            (
                ChunkingEvaluator._compute_recall(query_sample["chunks_idx"], ranked_chunks_idxes[i], k),
                ChunkingEvaluator._compute_ndcg(
                    query_sample["chunks_idx"],
                    ranked_chunks_idxes[i],
                    ranked_chunks_scores[i],
                    k,
                )
            )
            for i, query_sample in enumerate(tqdm(queries_dataset))
        ]
        recalls, ndcgs = zip(*metrics)

        return recalls, ndcgs
