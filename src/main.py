from dotenv import load_dotenv

from src.config.config import read_config
from src.utils import get_pdf_filepaths, get_pipeline
from src.evaluation.parsing_evaluator import ParsingEvaluator


load_dotenv()


def main():
    """Runs an experiment"""
    config = read_config()
    filepaths = get_pdf_filepaths(config.FILES_DIR)
    pipeline = get_pipeline(config.PACKAGE_TO_TEST)
    if config.DEVICE is not None:
        pipeline.set_device(config.DEVICE)
    config.DEVICE = pipeline.device  # use the config device enabled by the pipeline

    evaluator = ParsingEvaluator(pipeline)
    evaluator.evaluate_parsing(filepaths)
    evaluator.push_results_to_hf(config.HF_REPO_FOR_RESULTS)


if __name__ == "__main__":
    main()
