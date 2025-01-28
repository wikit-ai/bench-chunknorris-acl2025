from datetime import datetime
import json
from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.config.config import read_config
from src.utils import get_pdf_filepaths, get_pipeline
from src.evaluation.evaluator import Evaluator
from src.chunkers.page_chunker import PageChunker
from src.chunkers.recursive_character_chunker import RecursiveCharacterChunker

load_dotenv()


def push_results(tested_package: str, hf_repo_id: str, filepaths_to_push: list[str]):
    """Pushes the results to huggingface"""
    api = HfApi()
    timestamp = str(datetime.now()).replace(":", "-").replace(".", "-")
    for filepath in filepaths_to_push:
        api.upload_file(
            path_or_fileobj=f"./{filepath}",
            path_in_repo=f"./{tested_package}/{timestamp}/{filepath}",
            repo_id=hf_repo_id,
            repo_type="dataset",
        )


def main():
    """Runs an experiment"""
    config = read_config()
    filepaths = get_pdf_filepaths(config.FILES_DIR)
    pipeline = get_pipeline(config.PACKAGE_TO_TEST)
    if config.DEVICE is not None:
        pipeline.set_device(config.DEVICE)
    config.DEVICE = pipeline.device # use the config device enabled by the pipeline

    chunkers = [None, PageChunker(), RecursiveCharacterChunker()]\
    if pipeline.default_chunker\
    else [PageChunker(), RecursiveCharacterChunker()]

    evaluator = Evaluator(pipeline, chunkers=chunkers)
    evaluator.evaluate(filepaths)


    with open("run_config.json", "w", encoding="utf8") as file:
        json.dump(config.model_dump(), file, indent=4, ensure_ascii=False)

    push_results(
        pipeline.__class__.__name__,
        config.HF_REPO_FOR_RESULTS,
        ["codecarbon_results.csv", "chunks.json", "parsing_data.json", "run_config.json"]
        )


if __name__ == "__main__":
    main()
