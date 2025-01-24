from torch import tensor
from torchmetrics import RetrievalMRR, RetrievalRecall
from rouge_score import rouge_scorer


from components import Chunk
from evaluation.dataset.schema import ReferenceSection


class RetrievalEvaluator:
    def __init__(self, top_k: int):
        if top_k <= 0:
            raise ValueError("Top k argument cannot be negative or zero.")
        self.top_k = top_k
        self.overlap = rouge_scorer.RougeScorer(["rouge1"])
        self.mrr = RetrievalMRR(top_k=top_k)
        self.recall = RetrievalRecall(top_k=top_k)

    def check_files_and_pages(
        self, source_file: str, source_page: int, list_chunks: list[Chunk]
    ) -> list[int]:
        return [
            n
            for n, chunk in enumerate(list_chunks)
            if chunk.file == source_file
            and chunk.page_start <= source_page <= chunk.page_end
        ]

    def check_chunks(
        self,
        text_section: str,
        list_chunks: list[Chunk],
        tolerance_overlap: float = 0.1,
    ) -> list[int]:
        scores_list = [
            self.overlap.score(chunk.text, text_section)["rouge1"].precision
            for chunk in list_chunks
        ]
        return [
            n for n, score in enumerate(scores_list) if score >= (1 - tolerance_overlap)
        ]

    def evaluate_retrieval(
        self,
        reference_section: ReferenceSection,
        list_chunks: list[Chunk],
        tolerance_overlap: float,
    ) -> int | list[None]:
        remaining_indices = self.check_files_and_pages(
            source_file=reference_section.source_file,
            source_page=reference_section.page,
            list_chunks=list_chunks,
        )

        correct_index = self.check_chunks(
            text_section=reference_section.passage,
            list_chunks=[list_chunks[i] for i in remaining_indices],
            tolerance_overlap=tolerance_overlap,
        )

        if len(correct_index) > 1:
            raise RuntimeError("More than 1 chunk contains the reference answer.")

        return correct_index

    def construct_true_labels(self, correct_index: int | None) -> list[int]:
        pred_labels = [False] * self.top_k
        if correct_index:
            pred_labels[correct_index] = True
        return pred_labels

    def __call__(
        self,
        reference_section: ReferenceSection,
        list_chunks: list[Chunk],
        list_cosine_similarity: list[float],
        tolerance_overlap: float = 0.1,
    ):
        correct_index = self.evaluate_retrieval(
            reference_section=reference_section,
            list_chunks=list_chunks,
            tolerance_overlap=tolerance_overlap,
        )

        true_labels = self.construct_true_labels(correct_index=correct_index)
        pred_similarity = list_cosine_similarity.copy()

        indexes = tensor([0] * self.top_k)
        recall_scores = self.recall(
            indexes=indexes,
            preds=tensor(pred_similarity),
            true_labels=tensor(true_labels),
        )
        mrr_scores = self.mrr(
            indexes=indexes,
            preds=tensor(pred_similarity),
            true_labels=tensor(true_labels),
        )

        return recall_scores.numpy()[0], mrr_scores.numpy()[0]
