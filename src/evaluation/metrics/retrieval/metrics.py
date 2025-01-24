import re
from typing import Literal

import numpy as np
import numpy.typing as npt
from unidecode import unidecode


class Recall:
    def __init__(self, top_k: int):
        """
        Initialize the Recall class with the top_k value.

        Args:
            top_k (int): The number of top elements to consider.
        """
        self.top_k = top_k

    def _integer_split_by_top_k(
        self, chunk_present_list: list[list[bool]]
    ) -> npt.NDArray[np.int64]:
        """
        Split the chunk_present_list into integers based on the top_k value.

        This method processes a list of lists of boolean values representing whether each chunk contains the reference section
        and returns an array of integers indicating the presence of chunks (1) within the top_k range or not (0).

        Args:
            chunk_present_list (list[list[bool]]): A list of lists of boolean values representing whether each chunk contains the reference section.

        Returns:
            npt.NDArray[np.int64]: An array of integers indicating the presence of chunks within the top_k range.

        Example:
            >>> recall = Recall(top_k=2)
            >>> recall._integer_split_by_top_k([[True, False, True], [False, True, False]])
            array([1, 1])
        """
        return np.array(
            [int(any(true_labels[: self.top_k])) for true_labels in chunk_present_list]
        )

    def __call__(self, chunk_present_list: list[list[bool]]) -> float:
        """
        Calculate the recall metric.

        This method computes the recall metric based on a list of lists of boolean values
        representing whether each chunk contains the reference section.

        Args:
            chunk_present_list (list[list[bool]]): A list of lists of boolean values representing whether each chunk contains the reference section.

        Returns:
            float: The recall metric value.
        """
        pred_labels = self._integer_split_by_top_k(
            chunk_present_list=chunk_present_list
        )
        return np.sum(pred_labels) / pred_labels.shape[0]


class MRR:
    def __init__(self, top_k: int):
        """
        Initialize the MRR class with the top_k value.

        Args:
            top_k (int): The number of top elements to consider.
        """
        self.top_k = top_k

    def _split_by_top_k(
        self, chunk_present_list: list[list[bool]]
    ) -> npt.NDArray[np.bool_]:
        """
        Split the chunk_present_list based on the top_k value.

        This method processes a list of boolean values representing whether each chunk contains the reference section
        and returns an array of boolean values indicating the presence of chunks within the top_k range.

        Args:
            chunk_present_list (list[list[bool]]): A list of lists of boolean values representing whether each chunk contains the reference section.
            top_k (int): The top_k value used to determine the range of chunks to consider.

        Returns:
            npt.NDArray[np.bool_]: An array of boolean values indicating the presence of chunks within the top_k range.
        """
        return np.array(
            [true_labels[: self.top_k] for true_labels in chunk_present_list]
        )

    def _reorder_list_bool(
        self, chunk_present_list: list[list[bool]]
    ) -> list[list[bool]]:
        """
        Reverses the order of boolean values in each sublist of the input list.

        This method takes a list of lists, where each sublist contains boolean values,
        and returns a new list of lists with the boolean values in each sublist reversed.

        Args:
            chunk_present_list (list[list[bool]]): A list of lists, where each sublist contains boolean values.

        Returns:
            list[list[bool]]: A new list of lists with the boolean values in each sublist reversed.

        """
        return [chunk[::-1] for chunk in chunk_present_list]

    def __call__(
        self,
        chunk_present_list: list[bool],
        rank_order: Literal["decreasing", "increasing"],
    ) -> float:
        """
        Calculate the Mean Reciprocal Rank (MRR) metric.

        This method computes the Mean Reciprocal Rank (MRR) metric based on a list of boolean values
        representing whether each chunk contains the reference section. The order of ranking can be
        specified as either 'decreasing' (N to 1) or 'increasing' (1 to N).

        Args:
            chunk_present_list (list[bool]): A list of boolean values representing whether each chunk contains the reference section.
            rank_order (Literal["decreasing", "increasing"]): The order of ranking to be used for calculating MRR. Increasing: 1 to N | Decreasing: N to 1.

        Returns:
            float: The MRR metric value.
        """
        if rank_order == "decreasing":
            chunk_present_list = self._reorder_list_bool(
                chunk_present_list=chunk_present_list
            )

        pred_labels: npt.NDArray[np.int64] = self._split_by_top_k(
            chunk_present_list=chunk_present_list
        ).astype(int)

        position_array = np.indices(pred_labels.shape)[1] + 1
        return np.average(np.sum(np.divide(pred_labels, position_array), axis=1))


class Rouge:
    def __init__(self, type_rouge: Literal["rouge_1"]):
        """
        Initialize the Rouge class with the type_rouge value.

        Args:
            type_rouge (Literal["rouge_1"]): The type of ROUGE metric to calculate.
        """
        self.type_rouge = type_rouge

    def _split_sentences(self, sentence: str) -> list[str]:
        """
        Split a sentence into a list of words.

        Args:
            sentence (str): The sentence to split.

        Returns:
            list[str]: A list of words split from the sentence.

        Example:
            >>> rouge = Rouge(type_rouge="rouge_1")
            >>> rouge._split_sentences("This is a sentence.")
            ['This', 'is', 'a', 'sentence.']
        """
        pattern = r"\w+"
        sentence_clean = unidecode(sentence).replace("\n", " ").lower().strip()
        return re.findall(pattern, sentence_clean)

    def _evaluate_overlapping_over_multiple(
        self,
        splitted_reference: npt.NDArray[np.str_],
        comparison: str,
    ) -> tuple[int, list[int], int]:
        """
        Evaluate the overlapping unigrams between a reference section and multiple comparison chunks.

        Args:
            splitted_reference (npt.NDArray[np.str_]): The split reference section as a numpy array of strings.
            comparison (str): A single comparison string.

        Returns:
            tuple[int, list[int], int]: A tuple containing:
                - overlap (int): The number of overlapping unigrams for each comparison chunk.
                - len_comparisons (list[int]): The length of each comparison chunk.
                - len_reference (int): The length of the reference section.
        """
        splitted_comparisons = [
            self._split_sentences(sentence=chunk) for chunk in comparison
        ]
        overlap = [
            len(np.intersect1d(splitted_reference, chunk_array))
            for chunk_array in splitted_comparisons
        ]
        len_comparisons = ([len(chunk) for chunk in splitted_comparisons],)
        len_reference = len(splitted_reference)

        return overlap, len_comparisons, len_reference

    def _evaluate_overlapping_over_one(
        self, splitted_reference: npt.NDArray[np.str_], comparison: str
    ) -> tuple[int, int, int]:
        """
        Evaluate the overlapping unigrams between a reference section and a single comparison chunk.

        Args:
            splitted_reference (npt.NDArray[np.str_]): The split reference section as a numpy array of strings.
            comparison (str): A single comparison string.

        Returns:
            tuple[int, int, int]: A tuple containing:
                - overlap (int): The number of overlapping unigrams.
                - len_comparisons (int): The length of the comparison chunk.
                - len_reference (int): The length of the reference section.
        """
        splitted_comparisons = self._split_sentences(sentence=comparison)
        overlap = len(np.intersect1d(splitted_reference, splitted_comparisons))
        len_comparisons = (len(splitted_comparisons),)
        len_reference = len(splitted_reference)

        return overlap, len_comparisons, len_reference

    def precision(
        self,
        reference_section: str,
        comparison: str | list[str],
    ) -> float:
        """
        Calculate the ROUGE precision by dividing the number of unigrams in the comparison that also
        appear in the reference section by the number of unigrams in the comparison.

        Args:
            reference_section (str): The reference section to compare against.
            comparison (str | list[str]): A single comparison string or a list of comparison strings to compare with the reference section.

        Returns:
            float: The precision value for the comparison(s).
        """
        splitted_reference = np.array(self._split_sentences(sentence=reference_section))
        if isinstance(comparison, str):
            overlap, len_comparison, _ = self._evaluate_overlapping_over_one(
                splitted_reference=splitted_reference, comparison=comparison
            )
        else:
            overlap, len_comparison, _ = self._evaluate_overlapping_over_multiple(
                splitted_reference=splitted_reference, comparison=comparison
            )

        return np.divide(overlap, len_comparison).tolist()

    def recall(
        self,
        reference_section: str,
        comparison: str | list[str],
    ) -> float:
        """
        Calculate the ROUGE recall by dividing the number of unigrams in the comparison that also
        appear in the reference section by the number of unigrams in the reference section.

        Args:
            reference_section (str): The reference section to compare against.
            comparison (str | list[str]): A single comparison string or a list of comparison strings to compare with the reference section.

        Returns:
            float: The recall value for the comparison(s).
        """
        splitted_reference = np.array(self._split_sentences(sentence=reference_section))
        if isinstance(comparison, str):
            overlap, _, len_reference = self._evaluate_overlapping_over_one(
                splitted_reference=splitted_reference, comparison=comparison
            )
        else:
            overlap, _, len_reference = self._evaluate_overlapping_over_multiple(
                splitted_reference=splitted_reference, comparison=comparison
            )

        return np.divide(overlap, len_reference).tolist()
