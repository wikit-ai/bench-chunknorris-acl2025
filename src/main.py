from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import HfApi

from .config.config import read_config
from .utils import get_pdf_filepaths, get_pipeline

load_dotenv()


def push_results(tested_package: str, hf_repo_id: str):
    """Pushes the results to huggingface"""
    api = HfApi()
    timestamp = datetime.now()
    api.upload_file(
        path_or_fileobj="./codecarbon_outputs/codecarbon_results.csv",
        path_in_repo=f"./codecarbon_raw/{tested_package}/{timestamp}.csv",
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

    for file in filepaths:
        pipeline.parse_file(file)
        chunks = pipeline.chunk()

    push_results(config.PACKAGE_TO_TEST, config.HF_REPO_FOR_RESULTS)


if __name__ == "__main__":
    main()
