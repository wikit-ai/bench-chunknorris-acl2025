import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.rerankers.abs_reranker import AbsReranker, ScoredChunk

class CrossEncoderReranker(AbsReranker):
    """Uses Cohere reranker models"""
    def __init__(
        self,
        model_hf_repo: str ="Alibaba-NLP/gte-multilingual-reranker-base",
        max_token_length:int = 512,
        batch_size:int = 32,
        n_to_rerank : int = 100,
        description : str = ""
        ):
        """Loads a reranker with transformer libray.

        Args:
            model_hf_repo (str, optional): Model HF repo, compatible with transformer library.
            n_to_rerank (int, optional): amount of chunks to rerank. Defaults to 100.
        """
        super().__init__(n_to_rerank=n_to_rerank, description=description)
        self.model_hf_repo = model_hf_repo
        self.max_token_length = max_token_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_hf_repo)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_hf_repo, trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    def rerank(self, query:str, chunks: list[str]) -> list[ScoredChunk]:
        """
        Args:
            query (str): the query.
            chunks (list[str]): the list of chunks to rerank.

        Returns:
            list[ScoredChunk]: the reranked chunks, ranked by descreasing score.
        """
        pairs = [[query,chunk] for chunk in chunks]
        scores_buffer = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i+self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=self.max_token_length).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores_buffer.extend(scores)

        return sorted([ScoredChunk(
            text=chunk,
            index=i,
            score=score
        ) for i, (chunk, score) in enumerate(zip(chunks, scores_buffer))],
        key=lambda x: x.score, reverse=True)