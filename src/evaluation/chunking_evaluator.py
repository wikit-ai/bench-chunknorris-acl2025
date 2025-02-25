from collections import defaultdict
from datetime import datetime
import json
import os

from typing import Any, Literal
import datasets
from huggingface_hub import HfApi
from tqdm import tqdm
from chromacache import ChromaCache
from chromacache.embedding_functions import SentenceTransformerEmbeddingFunction

from src.pipelines.abs_pipeline import AbsPipeline
from src.components import Chunk
from src.chunkers.abs_chunker import AbstractChunker
from src.evaluation.chunking_evaluator_utils import chunks_to_dataset, run_scoring


class ChunkingEvaluator:
    """Meant to run an evaluation on a parser and set of chunkers."""

    timestamp: str

    def __init__(
        self,
        pipeline: AbsPipeline,
        chunkers: list[AbstractChunker | None],
        results_dir: str = "./results",
        sentence_transformer_hf_repo: str = "BAAI/bge-small-en-v1.5",
    ):
        """Instanciate an evaluator.

        Args:
            pipeline (AbsPipeline): the pipeline to be tested. Must inherit from AbsPipeline
            chunkers (None | list[AbstractChunker]): list of chunker to be tester.
                NOTE: If "None" is passed in the list, then the default_chunker of the pipeline will be used.
            sentence_transformer_hf_repo (str): the HF repo of a model compatible with SentenceTransformer.
                Used to embed chunks and queries to compute metrics. Defaults to BAAI/bge-small-en-v1.5.
        """
        self.pipeline = pipeline
        self.chunkers = chunkers
        self.results_dir = self._set_result_dir(results_dir)
        self.sentence_transformer_hf_repo = sentence_transformer_hf_repo

    def _set_result_dir(self, results_dir) -> str:
        """Set the directory in which results will be stored"""
        self.timestamp = str(datetime.now()).replace(":", "-").replace(".", "-")
        results_dir = os.path.join(results_dir, "chunking", self.timestamp)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

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
        pdf_filepaths: list[str],
        queries_dataset: datasets.Dataset | None,
    ):
        """Runs an experiment.

        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing to pdf files of the evaluation dataset.
            queries_dataset (datasets.Dataset | None): the dataset a queries to use for evaluation.
                NOTE: The dataset schema (column names, types, and overall structure) must be equivalent
                to Wikit's PIRE dataset. For more info see : https://huggingface.co/datasets/Wikit/PIRE.
                If None, Wikit/PIRE dataset will be used, assuming the pdf filepaths point to the PDF files from this dataset.
        """
        _ = self.get_chunks(pdf_filepaths)
        chunks_datadict = chunks_to_dataset(
            os.path.join(self.results_dir, "chunks.json")
        )
        if queries_dataset is None:
            queries_dataset = ChunkingEvaluator._load_eval_dataset(
                eval_split="chunk.multi"
            )
            self.run_chunking_evaluation(chunks_datadict, queries_dataset)
            queries_dataset = ChunkingEvaluator._load_eval_dataset(
                eval_split="chunk.single"
            )
            self.run_chunking_evaluation(chunks_datadict, queries_dataset)
        else:
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
                chunks_dict[chunker_name].extend(chunks)

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
    def _load_eval_dataset(
        eval_dataset_repo: str = "Wikit/PIRE",
        eval_split: Literal["chunk.multi", "chunk.single"] = "chunk.multi",
    ):
        """Loads the evaluation dataset.

        Args:
            eval_dataset_repo (str, optional): the repo of the dataset to load. Defaults to "Wikit/PIRE".
            eval_split (Literal[&quot;chunks.single&quot;, &quot;chunks.multi&quot;], optional):
                The split of the evaluation dataset. Defaults to "chunks.multi". For more info, see https://huggingface.co/datasets/Wikit/PIRE
        """
        queries = datasets.load_dataset(eval_dataset_repo)[eval_split]

        return queries

    def run_chunking_evaluation(
        self,
        chunks_datadict: datasets.DatasetDict,
        queries_dataset: datasets.Dataset,
        sentence_transformer_hf_repo: str | None = None,
    ):
        """Runs an evaluation to assess the chunking performance.

        Args:
            chunks_datadict (datasets.DatasetDict): a DatasetDict of the chunks, where:
                - each split has name as: parsername__chunkername.
                - a column name "text" contains the chunks' text.
            queries_dataset (datasets.Dataset): the dataset a queries to use for evaluation.
                NOTE: The dataset schema (column names, types, and overall structure) must be equivalent
                to Wikit's PIRE dataset. For more info see : https://huggingface.co/datasets/Wikit/PIRE
            sentence_transformer_hf_repo (str): the HF repo of a model compatible with SentenceTransformer.
                Used to embed chunks and queries to compute metrics. Defaults to None, which leads to using
                the model passed to __init__.
        """
        if sentence_transformer_hf_repo is None:
            sentence_transformer_hf_repo = self.sentence_transformer_hf_repo
        model = ChunkingEvaluator._get_model(sentence_transformer_hf_repo)

        queries_dataset = queries_dataset.add_column(
            "emb", model.encode(queries_dataset["query"])
        )

        results: list[dict[str, str | float]] = []
        for split in chunks_datadict.keys():
            parser_name, chunker_name = split.split("__")
            chunks = chunks_datadict[split]
            chunks = chunks.add_column("emb", model.encode(chunks["text"]))
            recall, ndcg = run_scoring(queries_dataset, chunks)

            results.append(
                {
                    "model": sentence_transformer_hf_repo,
                    "parser": parser_name,
                    "chunker": chunker_name,
                    "recall": recall,
                    "ndcg": ndcg,
                    "eval_split": str(queries_dataset.split),
                }
            )

        self.save_as_json(results, f"{str(queries_dataset.split)}_results.json")

    @staticmethod
    def _get_model(sentence_transformer_hf_repo: str) -> ChromaCache:
        return ChromaCache(
            SentenceTransformerEmbeddingFunction(sentence_transformer_hf_repo),
            save_embbedings=True,
            path_to_chromadb="./ChromaDB",
            batch_size=32,
        )

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
