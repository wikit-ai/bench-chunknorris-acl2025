from typing import Literal

import numpy as np

from src.components import Chunk, GeneratedAnswer, ReferenceSection

from src.evaluation.metrics.generation.schema import GenerationMetrics
from src.evaluation.metrics.retrieval.metrics import MRR, Rouge, Recall
from src.evaluation.metrics.retrieval.schema import RetrievalMetrics


class GenerationEvaluator:
    def __init__(self):
        self.rouge = Rouge(type_rouge="rouge_1")

    def evaluate_overlapping(
        self,
        list_references: list[ReferenceSection],
        list_generated_answers: list[GeneratedAnswer],
    ) -> list[float]:
        """
        Evaluate the overlapping between generated answers and reference passages using ROUGE precision.

        Args:
            list_references (list[ReferenceSection]): A list of reference sections, each containing a target passage.
            list_generated_answers (list[GeneratedAnswer]): A list of generated answers, each containing a generation.

        Returns:
            list[float]: A list of ROUGE precision scores for each pair of generated answer and reference passage.
        """
        list_generation = [answer.generation for answer in list_generated_answers]
        list_references_passage = [
            reference.target_passage for reference in list_references
        ]

        return [
            self.rouge.precision(reference_section=reference, comparison=generation)
            for reference, generation in zip(list_generation, list_references_passage)
        ]

    def __call__(
        self,
        list_references: list[ReferenceSection],
        list_generated_answers: list[GeneratedAnswer],
    ) -> GenerationMetrics:
        """
        Evaluate the generated answers and return a GenerationMetrics object.

        Args:
            list_references (list[ReferenceSection]): A list of reference sections, each containing a target passage.
            list_generated_answers (list[GeneratedAnswer]): A list of generated answers, each containing a generation.

        Returns:
            GenerationMetrics: An object containing the semantic similarity and average token counts of the generated answers.
        """
        return GenerationMetrics(
            rouge_precision=self.evaluate_overlapping(
                list_references=list_references,
                list_generated_answers=list_generated_answers,
            ),
            avg_generation_token_counts=np.mean(
                [len(answer.generation) for answer in list_generated_answers]
            ),
        )


class RetrievalEvaluator:
    """
    A class to evaluate the retrieval performance.

    Attributes:
        top_k (int): The number of top elements to consider.
        rouge (Rouge): An instance of the Rouge class for calculating ROUGE metrics.
        mrr (Recall): An instance of the Recall class for calculating MRR metrics.
        recall (MRR): An instance of the MRR class for calculating Recall metrics.
    """

    def __init__(self, top_k: int):
        """
        Initialize the RetrievalEvaluator class with the top_k value.

        Args:
            top_k (int): The number of top elements to consider. Must be greater than zero.

        Raises:
            ValueError: If top_k is negative or zero.
        """
        if top_k <= 0:
            raise ValueError("Top k argument cannot be negative or zero.")
        self.top_k = top_k
        self.rouge = Rouge(type_rouge="rouge_1")
        self.recall = Recall(top_k=top_k)
        self.mrr = MRR(top_k=top_k)

    def check_files_and_pages(
        self, source_file: str, source_page: int, list_chunks: list[Chunk]
    ) -> list[int]:
        """
        Check which chunks match the reference's source file and page.

        Args:
            source_file (str): The source file to match.
            source_page (int): The source page to match.
            list_chunks (list[Chunk]): A list of chunks to check.

        Returns:
            list[int]: A list of indices of chunks that match the source file and page.
        """
        return [
            n
            for n, chunk in enumerate(list_chunks)
            if chunk.source_file == source_file
            and (
                (chunk.page_start + 1) <= source_page <= (chunk.page_end + 1)
                or (chunk.page_start) <= source_page <= (chunk.page_end)
            )
        ]

    def check_chunks(
        self,
        text_section: str,
        list_chunks: list[Chunk],
        remaining_indices: list[int],
    ) -> list[int]:
        """
        Check which chunks have a ROUGE precision score above the tolerance overlap (0.7).

        Args:
            text_section (str): The reference text section to compare against.
            list_chunks (list[Chunk]): A list of chunks to check.
            remaining_indices (list[int]): A list of indices corresponding to the chunks.

        Returns:
            list[int]: A list of indices of chunks that have a ROUGE precision score above the tolerance overlap.
        """
        list_chunks_passages = [chunk.text for chunk in list_chunks]
        scores_list = self.rouge.recall(
            comparison=list_chunks_passages, reference_section=text_section
        )
        return [
            ind
            for ind, score in zip(remaining_indices, scores_list)
            if score >= 0.7 and score == max(scores_list)
        ]

    def get_index_chunk(
        self, correct_chunk: Chunk, list_chunks: list[Chunk]
    ) -> list[int]:
        """
        Get the indices of the correct chunk within a list of chunks.
        Args:
            correct_chunk (Chunk): The chunk to find within the list of chunks.
            list_chunks (list[Chunk]): The list of chunks to search within.

        Returns:
            list[int]: A list of indices where the correct chunk is found.
        """
        return [
            n
            for n, potential_chunk in enumerate(list_chunks)
            if potential_chunk == correct_chunk[0]
        ]

    def evaluate_retrieval(
        self,
        reference_section: ReferenceSection,
        list_chunks: list[Chunk],
    ) -> int | None:
        """
        Evaluate the retrieval of chunks against a reference section.

        Args:
            reference_section (ReferenceSection): The reference section to evaluate against.
            list_chunks (list[Chunk]): A list of chunks to evaluate.

        Returns:
            int | None: The index of the correct chunk, or None if not found.

        Raises:
            RuntimeError: If more than one chunk contains the reference answer.
        """
        remaining_indices = self.check_files_and_pages(
            source_file=reference_section.source_file,
            source_page=reference_section.target_page,
            list_chunks=list_chunks,
        )
        correct_chunk = self.check_chunks(
            text_section=reference_section.target_passage,
            list_chunks=[list_chunks[i] for i in remaining_indices],
            remaining_indices=remaining_indices,
        )

        if correct_chunk:
            return correct_chunk[0]
        return None

    def construct_reference_labels(self, correct_index: int | None) -> list[bool]:
        """
        Construct a list of reference labels based on the correct index.

        Args:
            correct_index (int | None): The index of the correct chunk, or None if not found.

        Returns:
            list[bool]: A list of boolean values indicating the presence of the correct chunk.
        """
        pred_labels = [False] * self.top_k
        if isinstance(correct_index, int):
            pred_labels[correct_index] = True
        return pred_labels

    def __call__(
        self,
        reference_list: list[ReferenceSection],
        list_chunks_per_question: list[list[Chunk]],
        rank_order: Literal["decreasing", "increasing"] = "increasing",
    ):
        """
        Evaluate the retrieval performance for a list of reference sections and chunks.

        Args:
            reference_list (list[ReferenceSection]): A list of reference sections to evaluate.
            list_chunks_per_question (list[list[Chunk]]): A list of lists of chunks to evaluate for each reference section.
            rank_order (Literal["decreasing", "increasing"], optional): The order of ranking for MRR calculation.
                Defaults to "increasing". Increasing: 1 to N | Decreasing: N to 1.

        Returns:
            RetrievalMetrics: The retrieval metrics including MRR, Recall, and a list of boolean values indicating the presence of the correct chunk.
        """
        correct_index_list = [
            self.evaluate_retrieval(
                reference_section=reference_section,
                list_chunks=list_chunks,
            )
            for reference_section, list_chunks in zip(
                reference_list, list_chunks_per_question
            )
        ]

        bool_labels_list = [
            self.construct_reference_labels(correct_index=correct_index)
            for correct_index in correct_index_list
        ]

        index_presence_chunks: list[int | None] = [
            bool_list.index(True) if any(bool_list) else None
            for bool_list in bool_labels_list
        ]

        return RetrievalMetrics(
            top_k=self.top_k,
            mrr=self.mrr(chunk_present_list=bool_labels_list, rank_order=rank_order),
            recall=self.recall(chunk_present_list=bool_labels_list),
            to_generate=[any(true_labels) for true_labels in bool_labels_list],
            index_presence_chunks=index_presence_chunks,
        )
