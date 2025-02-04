import json
import os
from typing import Any
from huggingface_hub import snapshot_download
import pandas as pd


def get_run_dirs(local_dir: str = "./results"):
    """Get the results from HF and aggregate results

    Args:
        local_dir (str, optional): the local directory where to clone the repo.
            Defaults to "./results".
    """
    # get the directory paths to all experiment runs
    pipeline_dirs = [dir for _, dirs, _ in os.walk(local_dir) for dir in dirs if "Pipeline" in dir]
    runs_dirs  = {
        os.path.join(local_dir, pipeline_dir, timestamp)
        for pipeline_dir in pipeline_dirs
        for _, timestamps, _ in os.walk(os.path.join(local_dir, pipeline_dir))
        for timestamp in timestamps
        }

    return runs_dirs


def get_results(hf_repo: str = "Wikit/pdf-parsing-bench-results", local_dir: str = "./results"):
    """Get the results from HF and aggregate results

    Args:
        hf_repo (str, optional): the hf repo from which to take data from.
            Defaults to "Wikit/pdf-parsing-bench-results".
        local_dir (str, optional): the local directory where to clone the repo.
            Defaults to "./results".
    """
    # get the data locally
    snapshot_download(repo_id=hf_repo, repo_type="dataset", local_dir=local_dir, ignore_patterns=["data/", "*.md"])

    runs_dirs = get_run_dirs(local_dir)
    runs_perfs, parsing_perfs = list(zip(*[aggregate_results_from_run(run_dir) for run_dir in runs_dirs]))
    runs_perfs = pd.DataFrame.from_records(runs_perfs)
    runs_perfs = runs_perfs.sort_values(by="timestamp")
    parsing_perfs = pd.concat(parsing_perfs)

    return runs_perfs, parsing_perfs


def aggregate_results_from_run(run_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Aggregate the results from a run.

    Args:
        run_dir (str): the path to the directory containing the results files
            of a specific run.

    Returns:
        tuple[dict[str, Any], pd.DataFrame]: the results.
    """
    # parse the code carbon info. Should be a csv of only 1 line
    carbon_info = pd.read_csv(os.path.join(run_dir, "codecarbon_results.csv"))
    carbon_info = carbon_info.iloc[-1].to_dict()

    with open(os.path.join(run_dir, "run_config.json"), "r", encoding="utf8") as f:
        config_info = json.load(f)

    parsing_info = pd.read_json(os.path.join(run_dir, "parsing_data.json"))
    parsing_info["pipeline"] = config_info["PACKAGE_TO_TEST"]
    parsing_info["device"] = config_info["DEVICE"]
    parsing_info["cpu_model"] = carbon_info["cpu_model"]
    parsing_info["run_id"] = carbon_info["run_id"]

    aggragate = carbon_info | config_info | {
        "avg_latency": float(parsing_info["parsing_latency"].mean()),
        "median_latency": float(parsing_info["parsing_latency"].median()),
        "std_latency": float(parsing_info["parsing_latency"].std()),
        "avg_cpu_load": float(parsing_info["cpu_load_percent"].mean()),
        "median_cpu_load": float(parsing_info["cpu_load_percent"].median()),
        "std_cpu_load": float(parsing_info["cpu_load_percent"].std()),
        "computing_config": get_config_name(carbon_info),
        "total_parsing_latency": float(parsing_info["parsing_latency"].sum())
        }

    return aggragate, parsing_info


def get_config_name(carbon_info):
    if "Windows" in carbon_info["os"] and "i7-13620H" in carbon_info["cpu_model"] and"RTX 4060 Laptop" in carbon_info["gpu_model"]:
        return "local"
    if "Linux" in carbon_info["os"] and "6226R" in carbon_info["cpu_model"] and "V100S" in carbon_info["gpu_model"]:
        return "ovhai"
    return "unknown"