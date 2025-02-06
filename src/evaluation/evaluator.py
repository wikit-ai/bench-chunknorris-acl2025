from collections import defaultdict
from datetime import datetime
import json
import os

import time
import pandas as pd
from huggingface_hub import HfApi
from psutil import cpu_percent
from tqdm import tqdm

from src.pipelines.abs_pipeline import AbsPipeline
from src.components import Chunk
from src.chunkers.abs_chunker import AbstractChunker


from src.utils import dynamic_track_emissions

class Evaluator():
    """Meant to run an evaluation"""
    def __init__(self, pipeline: AbsPipeline, chunkers: None | list[AbstractChunker] = None):
        """Instanciate an evaluator.

        Args:
            pipeline (AbsPipeline): the pipeline to be tested. Must inherit from AbsPipeline
            chunkers (None | list[AbstractChunker]): list of chunker to be tester.
                NOTE: If "None" is passed in the list, then the default_chunker of the pipeline will be used.
        """
        self.pipeline = pipeline
        self.chunkers = chunkers or [None]

    @dynamic_track_emissions
    def evaluate_file_parsing(self, pdf_filepaths: list[str]) -> dict[str, list[Chunk]]:
        """Considering the pipeline and the list of chunkers provided to the evaluator,
        gets the chunks obtained from the file for each chunker.
        
        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing to pdf to submit to test.

        Returns:
            dict[str, list[Chunk]] : a dict with the chunker's name as key
                and the list chunks of all documents as value.
        """
        parsing_data: list[dict] = []
        # parse the file
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

        with open("parsing_data.json", "w", encoding="utf8") as file:
            json.dump(parsing_data, file, indent=4, ensure_ascii=False)


    def get_chunks(self, pdf_filepaths: list[str], chunkers: list[AbstractChunker]) -> dict[str, list[Chunk]]:
        """Considering the pipeline and the list of chunkers provided to the evaluator,
        gets the chunks obtained from the file for each chunker.
        
        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing to pdf to submit to test.
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

        dumped_chunks = {chunker: [chunk.model_dump() for chunk in chunks] for chunker, chunks in chunks_dict.items()}
        pipeline_name = self.pipeline.__class__.__name__
        with open(f"{pipeline_name}_chunks.json", "w", encoding="utf8") as file:
            json.dump(dumped_chunks, file, indent=4, ensure_ascii=False)

        return chunks_dict

    def evaluate(self, pdf_filepaths: list[str]):
        """Runs an experiment
        
        Args:
            pdf_filepaths (list[str]): the list of filepaths pointing to pdf to submit to test.

        """
        chunks_dict = self.get_chunks(pdf_filepaths, self.chunkers)


    def process_codecarbon_results(self):
        """Post process codecarbon's results"""

        codecarbon_results_path = "./codecarbon_results.csv"
        data = pd.read_csv(codecarbon_results_path)
        return


    def push_results(self, tested_package: str, hf_repo_id: str):
        """Pushes the results to huggingface"""
        api = HfApi()
        timestamp = datetime.now()
        api.upload_file(
            path_or_fileobj="./codecarbon_results.csv",
            path_in_repo=f"./codecarbon_raw/{tested_package}/{timestamp}.csv",
            repo_id=hf_repo_id,
            repo_type="dataset",
        )
