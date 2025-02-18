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

class ChunkingEvaluator():
    """Meant to run an evaluation on a parser and set of chunkers."""
    timestamp : str
    def __init__(
            self,
            pipeline: AbsPipeline,
            chunkers: None | list[AbstractChunker] = None,
            results_dir : str = "./results"
            ):
        """Instanciate an evaluator.

        Args:
            pipeline (AbsPipeline): the pipeline to be tested. Must inherit from AbsPipeline
            chunkers (None | list[AbstractChunker]): list of chunker to be tester.
                NOTE: If "None" is passed in the list, then the default_chunker of the pipeline will be used.
        """
        self.pipeline = pipeline
        self.chunkers = chunkers or [None]
        self.results_dir = self._set_result_dir(results_dir)


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
        with open(os.path.join(
            self.results_dir, filename), "w", encoding="utf8") as file:
            json.dump(dict_to_save, file, indent=4, ensure_ascii=False)


    def evaluate_chunking(self, pdf_filepaths: list[str]):
        """Runs an experiment.

        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing to pdf to submit to test.
            eval_dataset_split (Literal["chunks.single", "chunks.multi"]): the split of the dataset to use for evaluation.
        """
        _ = self.get_chunks(pdf_filepaths, self.chunkers)
        chunks_datadict = chunks_to_dataset(os.path.join(self.results_dir, "chunks.json"))
        self.run_chunking_evaluation(chunks_datadict, "chunk.single")
        self.run_chunking_evaluation(chunks_datadict, "chunk.multi")


    def get_chunks(self, pdf_filepaths: list[str], chunkers: list[AbstractChunker]) -> dict[str, list[Chunk]]:
        """Considering the pipeline and the list of chunkers provided to the evaluator,
        gets the chunks obtained for each files and for each chunker.
        
        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing the pdfs of the retrieval dataset.
                They can be found here: https://huggingface.co/datasets/Wikit/PIRE/tree/main
            chunkers (list[AbsChunker]): the list of chunkers to get the chunks from. 
                NOTE : Pass "None" in the list of chunkers to also use the pipeline's default chunker.

        Returns:
            dict[str, list[Chunk]] : a dict with the chunker's name as key
                and the list chunks of all documents as value.
        """
        chunks_dict : dict[str, list[Chunk]] = defaultdict(list)
        # parse the file
        for filepath in tqdm(pdf_filepaths):
            self.pipeline.parse_file(filepath)
            # use the result of the parsing to chunk with the chunkers
            for chunker in chunkers:
                self.pipeline.external_chunker = chunker
                chunker_name = "Default" if chunker is None else self.pipeline.external_chunker.__class__.__name__
                chunks = self.pipeline.chunk()
                chunks_dict[chunker_name].extend(chunks)

        pipeline_name = self.pipeline.__class__.__name__
        dumped_chunks = {
            pipeline_name : {
                chunker: [chunk.model_dump() for chunk in chunks] for chunker, chunks in chunks_dict.items()
                }}
        self.save_as_json(dumped_chunks, "chunks.json")

        return chunks_dict


    def run_chunking_evaluation(
        self,
        chunks_datadict : datasets.DatasetDict,
        eval_dataset_split: Literal["chunk.single", "chunk.multi"] = "chunk.multi",
        sentence_transformer_hf_repo: str = "BAAI/bge-small-en-v1.5",
        ):
        """Runs an evaluation of the chunking performance.

        Args:
            chunks_datadict (datasets.DatasetDict): a dataset dict of the chunks, where each split has name as: parsername__chunkername
            eval_dataset_split (Literal[&quot;chunks.single&quot;, &quot;chunks.multi&quot;], optional): 
                The split of the evaluation dataset. Defaults to "chunks.multi". For more info, see https://huggingface.co/datasets/Wikit/PIRE
            sentence_transformer_hf_repo (str): the HF repo of a model compatible with SentenceTransformer.
                Used to embed chunks and queries to compute metrics.
        """
        model = ChunkingEvaluator._get_model(sentence_transformer_hf_repo)
        queries = datasets.load_dataset("Wikit/PIRE")[eval_dataset_split]
        queries = queries.add_column("emb", model.encode(queries["query"]))

        results: list[dict[str, str | float]]= []
        for split in chunks_datadict.keys():
            parser_name, chunker_name = split.split("__")
            chunks = chunks_datadict[split]
            chunks = chunks.add_column("emb", model.encode(chunks["text"]))
            recall, ndcg = run_scoring(queries, chunks)

            results.append({
                "model": sentence_transformer_hf_repo,
                "parser": parser_name,
                "chunker": chunker_name,
                "recall": recall,
                "ndcg": ndcg,
                "eval_split": eval_dataset_split
            })

        self.save_as_json(results, f"{eval_dataset_split}_results.json")


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
