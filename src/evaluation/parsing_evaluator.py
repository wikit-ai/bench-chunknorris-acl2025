from datetime import datetime
import json
import os
import time
from typing import Any

from psutil import cpu_percent
from tqdm import tqdm
from huggingface_hub import HfApi

from src.components import Chunk
from src.pipelines.abs_pipeline import AbsPipeline
from src.utils import dynamic_track_emissions
from src.evaluation.parsing_evaluator_utils import aggregate_results_from_run


class ParsingEvaluator():
    """Meant to run an evaluation on a parser."""
    timestamp : str # timestamp of the experiment

    def __init__(
            self,
            pipeline: AbsPipeline,
            results_dir : str = "./results"
            ):
        """Instanciate an evaluator.

        Args:
            pipeline (AbsPipeline): the pipeline to be tested. Must inherit from AbsPipeline
        """
        self.pipeline = pipeline
        self.results_dir = self._set_result_dir(results_dir)
        self.save_config()

    def _set_result_dir(self, results_dir) -> str:
        """Set the directory in which results will be stored"""
        self.timestamp = str(datetime.now()).replace(":", "-").replace(".", "-")
        results_dir = os.path.join(results_dir, "parsing", self.timestamp)
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

    def save_config(self):
        """Saves information about the run config"""
        tmp = {
            "pipeline": self.pipeline.__class__.__name__,
            "device": self.pipeline.device
            }
        self.save_as_json(tmp, "run_config.json")


    def evaluate_parsing(self, pdf_filepaths: list[str]):
        """Parses the pdf files and processes the results obtained
        
        Args:
            pdf_filepaths (list[str]) : the paths to the pdf files.
        """
        self.run_parsing(pdf_filepaths)
        processed_results, _ = aggregate_results_from_run(self.results_dir)
        self.save_as_json(processed_results, "processed_results.json")


    @dynamic_track_emissions
    def run_parsing(self, pdf_filepaths: list[str]) -> dict[str, list[Chunk]]:
        """Considering the pipeline and the list of chunkers provided to the evaluator,
        gets the chunks obtained from the file for each chunker.
        
        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing to pdf to submit to test.

        Returns:
            dict[str, list[Chunk]] : a dict with the chunker's name as key
                and the list chunks of all documents as value.
        """
        parsing_data: list[dict] = []
        for filepath in tqdm(pdf_filepaths):
            cpu_percent()
            start_time = time.perf_counter()
            self.pipeline.parse_file(filepath)
            parsing_data.append(
                {
                    "filename": os.path.basename(filepath),
                    "cpu_load_percent": cpu_percent(),
                    "parsing_latency": time.perf_counter() - start_time
                }
            )

        self.save_as_json(parsing_data, "cpuload_latencies.json")


    def push_results_to_hf(self, hf_repo_id: str):
        """Pushes the results to huggingface"""
        api = HfApi()
        for filepath in os.listdir(self.results_dir):
            api.upload_file(
                path_or_fileobj=os.path.join(self.results_dir, filepath),
                path_in_repo=os.path.join("parsing", self.timestamp, filepath),
                repo_id=hf_repo_id,
                repo_type="dataset",
            )
