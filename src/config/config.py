import os
from typing import Literal
import yaml

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Handles the configuration parameters."""

    FILES_DIR: str = Field(
        description="Path to the directory where pdf files are located."
    )
    PACKAGE_TO_TEST: str = Field(
        default="chunknorris",
        description="The package to use for benchmarking. Must be matched in get_pipeline() function of main.py",
    )
    DEVICE: Literal["cpu", "cuda"] | None = Field(
        default=None,
        description="The device to use. If None, will default to the package's default device.",
    )
    HF_REPO_FOR_RESULTS: str = Field(
        default="Wikit/pdf-parsing-bench-results",
        description="Where the output results will be loaded.",
    )
    COUNTRY_ISO_CODE: str = Field(
        default="FRA",
        description="3-letter country code. Used by codecarbon to kgCO2eq emissions resulting of energy production.",
    )


def read_config(filepath: str = "experiment.config.yml") -> Config:
    """Reads a config file

    Args:
        filepath (str, optional): the path to the .yml file. Defaults to 'config.yml'.

    Returns:
        dict[str, Any]: a dict wit all variables
    """
    with open(filepath, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Override default values of config.yaml with variables specified using --env
    for env_var in config:
        config[env_var] = os.getenv(env_var, None) or config[env_var]

    # just set COUNTRY_ISO_CODE as it is need by the codecarbon decorator
    os.environ["COUNTRY_ISO_CODE"] = config["COUNTRY_ISO_CODE"]

    return Config(**config)
