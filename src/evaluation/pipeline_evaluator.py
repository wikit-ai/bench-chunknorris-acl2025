import json
import os
from typing import Literal

from datasets import load_dataset, Dataset
import numpy as np
import numpy.typing as npt

from src.components import Chunk, ReferenceSection
from src.evaluation.metrics.evaluator import RetrievalEvaluator
from src.evaluation.sentence_embeddings.utils import retrieve_top_k_chunks
from src.evaluation.sentence_embeddings.model import model


class PipelinePerformanceEvaluator:
    """
    A class to evaluate the performance of a pipeline for text retrieval tasks.
    """

    def __init__(self):
        self.pipeline = None
        self.chunker = None
        self.embedding_name = None

    def load_pipeline_components(
        self,
        pipeline: Literal[
            "base", "chunknorris", "docling", "marker", "openparsecpu", "openparsegpu"
        ],
        chunker: Literal["Default", "PageChunker", "RecursiveCharacterChunker"],
    ) -> tuple[int, list[Chunk], list[ReferenceSection], npt.NDArray[np.float32]]:
        """
        Load the components of a pipeline, including chunks and embeddings, based on the specified pipeline, chunker, and embedding type.

        Args:
            pipeline (Literal["base", "chunknorris", "docling", "marker", "openparsecpu", "openparsegpu"]):
                The name of the pipeline to use.
            chunker (Literal["Default", "PageChunker", "RecursiveCharacterChunker"]):
                The type of chunker to use.
        Returns:
            tuple[int, list[Chunk], list[ReferenceSection], npt.NDArray[np.float32]]:
                A tuple containing:
                - len_empty_strings (int): The number of empty strings in the dataset.
                - list_chunks (list[Chunk]): A list of chunks extracted from the dataset.
                - list_reference_sections (list[ReferenceSection]): A list of reference sections.
                - list_embeddings (npt.NDArray[np.float32]): A numpy array containing the embeddings for the chunks.

        Raises:
            ValueError: If the specified combination of pipeline and chunker does not exist.
        """
        dataset_dict = load_dataset("Wikit/pdf-parsing-bench-results")
        try:
            dataset = dataset_dict[pipeline + "_" + chunker]
        except Exception as _:
            raise ValueError(
                f"This combination of pipeline/chunker ({pipeline}_{chunker}) does not exist."
            ) from _

        len_empty_strings, clean_dataset = self.cleaning_empty_strings(
            dataset=dataset, chunker=chunker
        )
        chunk_dict = [
            {
                "text": item["text"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "source_file": item["source_file"],
            }
            for item in clean_dataset
        ]
        list_chunks = [Chunk(**chunk) for chunk in chunk_dict]

        if os.getenv("EMBEDDING") in [
            "snowflake-arctic-embed-m-v2.0",
            "potion-retrieval-32M",
        ]:
            self.embedding_name = self._get_embedding_column()
            list_embeddings = np.array(clean_dataset[self.embedding_name])
        else:
            list_embeddings = model.get_embeddings(clean_dataset["text"])
            self.embedding_name = "emb_bge_base"

        return (
            len_empty_strings,
            list_chunks,
            self._get_reference_section(),
            list_embeddings,
        )

    @staticmethod
    def check_if_space(string: str):
        """
        Check if the given string is not empty and contains at least 10 non-space characters.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string is not empty and contains at least 10 non-space characters, False otherwise.
        """
        return not string.isspace() and len(string) >= 10

    def cleaning_empty_strings(
        self, dataset: Dataset, chunker: str
    ) -> tuple[int, Dataset]:
        """
        Clean the dataset by removing empty strings and strings that do not meet the criteria.

        Args:
            dataset (Dataset): The dataset to clean.
            chunker (str): The type of chunker used.

        Returns:
            tuple[int, Dataset]: A tuple containing:
                - len_empty_string (int): The number of empty strings removed from the dataset.
                - clean_dataset (Dataset): The cleaned dataset.
        """
        len_full_dataset = len(dataset["text"])
        clean_dataset = dataset.filter(self.check_if_space, input_columns="text")
        len_empty_string = len_full_dataset - len(clean_dataset["text"])
        if chunker != "Default":
            return len_empty_string, clean_dataset
        return len_empty_string, dataset

    def _get_embedding_column(self) -> str:
        """
        Get the embedding column name based on the type of embedding.

        Args:
            embedding (Literal["contextual_embedding", "static_embedding"]): The type of embedding.

        Returns:
            str: The name of the embedding column corresponding to the specified embedding type.
        """
        dict_models = {
            "snowflake-arctic-embed-m-v2.0": "emb_sf_m_v2",
            "potion-retrieval-32M": "emb_potion_r32M",
        }
        return dict_models[os.getenv("EMBEDDING")]

    def _get_reference_section(self) -> list[ReferenceSection]:
        """
        Load and return a list of reference sections from a JSON file.

        Returns:
            list[ReferenceSection]: A list of ReferenceSection objects loaded from the JSON file.
        """
        with open(r"dataset\storage\dataset.json", encoding="utf-8") as f:
            json_reference = json.load(f)

        return [ReferenceSection(**ref) for ref in json_reference]

    def save_json_results(
        self,
        len_empty_strings: int,
        list_references: list[ReferenceSection],
        list_top_chunks: list[list[Chunk]],
        results_retrieval: RetrievalEvaluator,
        chunker: Literal["Default", "PageChunker", "RecursiveCharacterChunker"],
        pipeline: Literal[
            "base", "chunknorris", "docling", "marker", "openparsecpu", "openparsegpu"
        ],
    ) -> None:
        """
        Save the retrieval results to a JSON file.

        Args:
            len_empty_strings (int): The number of empty strings in the dataset.
            list_references (list[ReferenceSection]): A list of reference sections.
            list_top_chunks (list[list[Chunk]]): A list of lists containing the top chunks for each reference section.
            results_retrieval (RetrievalEvaluator): The results from the retrieval evaluator.
            chunker (Literal["Default", "PageChunker", "RecursiveCharacterChunker"]): The type of chunker used.
            pipeline (Literal["base", "chunknorris", "docling", "marker", "openparsecpu", "openparsegpu"]):
                The name of the pipeline used.

        Returns:
            None
        """
        references_dict = [reference.model_dump() for reference in list_references]

        top_chunks_dict = [
            [chunk.model_dump() for chunk in chunks] for chunks in list_top_chunks
        ]

        results = [results_retrieval.dict()]
        results[0]["count_empty_strings"] = len_empty_strings
        for ref, bool_presence, ind_presence in zip(
            references_dict,
            results[0]["to_generate"],
            results[0]["index_presence_chunks"],
        ):
            ref["chunk_present"] = {"presence": bool_presence, "index": ind_presence}

        results[0].pop("to_generate")
        results[0].pop("index_presence_chunks")
        results.append(
            [
                {"references": reference, "top_chunk": top_chunk}
                for reference, top_chunk in zip(references_dict, top_chunks_dict)
            ]
        )
        directory = f"results_retrieval/{self.embedding_name}"
        os.makedirs(directory, exist_ok=True)
        with open(
            f"results_retrieval/{self.embedding_name}/retrieval_{chunker.lower()}_{pipeline.lower()}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def evaluate_retrieval(
        self,
        top_k: int,
        pipeline: Literal[
            "base", "chunknorris", "docling", "marker", "openparsecpu", "openparsegpu"
        ],
        chunker: Literal["Default", "PageChunker", "RecursiveCharacterChunker"],
    ) -> RetrievalEvaluator:
        """
        Evaluate the retrieval performance of the pipeline and save the results.

        Args:
            top_k (int): The number of top chunks to retrieve for each reference passage.
            pipeline (Literal["base", "chunknorris", "docling", "marker", "openparsecpu", "openparsegpu"]):
                The name of the pipeline to use.
            chunker (Literal["Default", "PageChunker", "RecursiveCharacterChunker"]): The type of chunker to use.
            embedding (Literal["contextual_embedding", "static_embedding"]): The type of embedding to use.

        Returns:
            RetrievalEvaluator: The results from the retrieval evaluator.
        """
        len_empty_strings, list_chunks, list_references, list_embeddings = (
            self.load_pipeline_components(pipeline=pipeline, chunker=chunker)
        )
        list_top_chunks: list[list[Chunk]] = retrieve_top_k_chunks(
            list_references=list_references,
            list_chunks=list_chunks,
            list_embeddings=list_embeddings,
            top_k=top_k,
        )
        evaluator = RetrievalEvaluator(top_k=top_k)
        results_retrieval = evaluator(
            reference_list=list_references, list_chunks_per_question=list_top_chunks
        )

        self.save_json_results(
            len_empty_strings=len_empty_strings,
            list_references=list_references,
            list_top_chunks=list_top_chunks,
            results_retrieval=results_retrieval,
            chunker=chunker,
            pipeline=pipeline,
        )

        return results_retrieval

    # def evaluate_generation(self):
    #     pass
