import json
import os
from typing import Any
from huggingface_hub import snapshot_download
import pandas as pd

COLS_TO_KEEP = [
    "timestamp",
    "run_id",
    "cpu_power",
    "gpu_power",
    "cpu_energy",
    "gpu_energy",
    "os",
    "cpu_model",
    "gpu_model",
    "PACKAGE_TO_TEST",
    "DEVICE",
    "avg_latency",
    "median_latency",
    "std_latency",
    "avg_latency_per_page",
    "std_latency_per_page",
    "avg_cpu_load",
    "computing_config",
    "total_parsing_latency",
]

N_PAGE_PER_DOC = {
    "arxiv1.pdf": 25,
    "arxiv2_taclccby4_license.pdf": 23,
    "arxiv3.pdf": 23,
    "arxiv4.pdf": 25,
    "arxiv5_ccby4license.pdf": 14,
    "00-80T-80.pdf": 434,
    "1001.0266.pdf": 11,
    "1001.0510.pdf": 7,
    "1001.0764.pdf": 16,
    "1001.0770.pdf": 5,
    "1001.0806.pdf": 6,
    "1001.0955.pdf": 5,
    "1001.2449.pdf": 6,
    "1001.2538.pdf": 5,
    "1001.2648.pdf": 4,
    "1001.2669.pdf": 33,
    "1001.2670.pdf": 4,
    "1002.2525.pdf": 13,
    "ASX_KCN_2013.pdf": 120,
    "ASX_MRM_2000.pdf": 72,
    "ASX_SEA_2014.pdf": 114,
    "ASX_STO_2004.pdf": 96,
    "basic-english-language-skills.PDF": 59,
    "Botswana-constitution.pdf": 59,
    "EN-Annex II - EU-OSHA websites, SM accounts and tools.pdf": 164,
    "EN-Draft FWC for services 0142.pdf": 49,
    "Excel Training Manual 1.pdf": 60,
    "Microscope Manual.pdf": 10,
    "NASDAQ_ATRI_2003.pdf": 32,
    "NASDAQ_EEFT_2000.pdf": 48,
    "NASDAQ_EMMS_2004.pdf": 8,
    "NASDAQ_FFIN_2002.pdf": 96,
    "NASDAQ_SHEN_2003.pdf": 60,
    "NYSE_AIT_2012.pdf": 48,
    "NYSE_CHK_2010.pdf": 48,
    "NYSE_GLW_2002.pdf": 12,
    "NYSE_HIG_2001.pdf": 40,
    "NYSE_HNI_2003.pdf": 64,
    "NYSE_HRL_2004.pdf": 13,
    "NYSE_JWN_2014.pdf": 96,
    "NYSE_MGM_2004.pdf": 82,
    "NYSE_RCI_2013.pdf": 132,
    "NYSE_RSG_2004.pdf": 106,
    "NYSE_SMFG_2011.pdf": 16,
    "OTC_NSANY_2004.pdf": 114,
    "PLAW-116publ30.pdf": 2,
    "sg246915.pdf": 440,
    "sg247938.pdf": 826,
    "sg248459.pdf": 270,
    "TSX_KMP_2013.pdf": 98,
    "uksi_20200438_en.pdf": 4,
    "uksi_20200471_en.pdf": 8,
    "uksi_20210538_en.pdf": 4,
    "uksi_20210582_en.pdf": 92,
    "tesla_form_10q.pdf": 49,
    "Wikimedia_Foundation_2024_Audited_Financial_Statements.pdf": 20,
    "BD-EN_calendrier-Lauzun-2024.pdf": 4,
    "infographic3.pdf": 1,
    "infographic5.pdf": 1,
    "Understanding_Creative_Commons_license_(infographic).pdf": 1,
    "legal1_opengouvernementlicense.pdf": 4,
    "legal2_opengouvernementlicense.pdf": 45,
    "legal4_opengouvernementlicense.pdf": 31,
    "legal5_eubiodiversity_cc4.pdf": 23,
    "office-pdf.pdf": 61,
    "serverless-core.pdf": 91,
    "welcome_to_word_template.pdf": 8,
    "news1.pdf": 1,
    "news2.pdf": 1,
    "news3.pdf": 5,
    "news4.pdf": 2,
    "MSTeams_QuickStartGuide_EN_Final_4.18.22.pdf": 6,
    "Publicdomain.pdf": 1,
    "Word QS.pdf": 4,
    "pubmed1.pdf": 13,
    "pubmed10.pdf": 22,
    "pubmed11.pdf": 27,
    "pubmed12.pdf": 11,
    "pubmed13.pdf": 11,
    "pubmed2.pdf": 14,
    "pubmed3.pdf": 25,
    "pubmed4.pdf": 18,
    "pubmed5.pdf": 14,
    "pubmed6_cc4.pdf": 13,
    "pubmed7_cc4.pdf": 33,
    "pubmed8.pdf": 12,
    "pubmed9.pdf": 14,
    "2023-Creative-Commons-Annual-Report-2-1.pdf": 12,
    "creative_common_ai.pdf": 22,
    "Open_Data_Report.pdf": 34,
    "6126797.pdf": 10,
    "CompostGuide.pdf": 8,
    "edp_s1_man_portal-version_4.3-user-manual_v1.0.pdf": 57,
    "maiis-user-manual.pdf": 50,
    "Protege5NewOWLPizzaTutorialV3.pdf": 91,
    "wikipedia1.pdf": 38,
    "wikipedia2.pdf": 28,
    "wikipedia3.pdf": 70,
    "wikipedia4.pdf": 25,
    "wikipedia5.pdf": 34,
}


def get_results(
    hf_repo: str = "Wikit/pdf-parsing-bench-results", local_dir: str = "./results"
):
    """Get the results from HF and aggregate results

    Args:
        hf_repo (str, optional): the hf repo from which to take data from.
            Defaults to "Wikit/pdf-parsing-bench-results".
        local_dir (str, optional): the local directory where to clone the repo.
            Defaults to "./results".
    """
    # get the data locally
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=["data/", "*.md", "*.git*", "results_retrieval*"],
    )

    runs_dirs = get_run_dirs(local_dir)
    runs_perfs, parsing_perfs = list(
        zip(*[aggregate_results_from_run(run_dir) for run_dir in runs_dirs])
    )
    runs_perfs = pd.DataFrame.from_records(runs_perfs)
    runs_perfs = runs_perfs.sort_values(by="timestamp")
    parsing_perfs = pd.concat(parsing_perfs)

    return runs_perfs, parsing_perfs


def get_run_dirs(local_dir: str = "./results/parsing"):
    """Get the results from HF and aggregate results

    Args:
        local_dir (str, optional): the local directory where to clone the repo.
            Defaults to "./results".
    """
    # get the directory paths to all experiment runs
    pipeline_dirs = [
        dir for _, dirs, _ in os.walk(local_dir) for dir in dirs if "Pipeline" in dir
    ]
    runs_dirs = {
        os.path.join(local_dir, pipeline_dir, timestamp)
        for pipeline_dir in pipeline_dirs
        for _, timestamps, _ in os.walk(os.path.join(local_dir, pipeline_dir))
        for timestamp in timestamps
    }

    return runs_dirs


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

    parsing_info = pd.read_json(os.path.join(run_dir, "cpuload_latencies.json"))
    parsing_info["pipeline"] = config_info["pipeline"]
    parsing_info["device"] = config_info["device"]
    parsing_info["cpu_model"] = carbon_info["cpu_model"]
    parsing_info["run_id"] = carbon_info["run_id"]
    parsing_info["n_doc_pages"] = [
        N_PAGE_PER_DOC[filename] for filename in parsing_info["filename"]
    ]

    aggragate = (
        carbon_info
        | config_info
        | {
            "avg_latency": float(parsing_info["parsing_latency"].mean()),
            "median_latency": float(parsing_info["parsing_latency"].median()),
            "std_latency": float(parsing_info["parsing_latency"].std()),
            "avg_latency_per_page": float(
                (parsing_info["parsing_latency"] / parsing_info["n_doc_pages"]).mean()
            ),
            "std_latency_per_page": float(
                (parsing_info["parsing_latency"] / parsing_info["n_doc_pages"]).std()
            ),
            "avg_cpu_load": float(parsing_info["cpu_load_percent"].mean()),
            "median_cpu_load": float(parsing_info["cpu_load_percent"].median()),
            "std_cpu_load": float(parsing_info["cpu_load_percent"].std()),
            "computing_config": get_config_name(carbon_info),
            "total_parsing_latency": float(parsing_info["parsing_latency"].sum()),
        }
    )
    aggragate["cpu_energy_Wh"] = (
        aggragate["total_parsing_latency"]
        / 3600
        * aggragate["cpu_power"]
        * aggragate["avg_cpu_load"]
        / 100
    )
    aggragate["gpu_energy_Wh"] = aggragate["gpu_energy"] / 1000

    return aggragate, parsing_info


def get_config_name(carbon_info):
    if (
        "Windows" in carbon_info["os"]
        and "i7-13620H" in carbon_info["cpu_model"]
        and "RTX 4060 Laptop" in carbon_info["gpu_model"]
    ):
        return "local"
    if (
        "Linux" in carbon_info["os"]
        and "6226R" in carbon_info["cpu_model"]
        and "V100S" in carbon_info["gpu_model"]
    ):
        return "ovhai"
    return "unknown"
