from mxbai_rerank import MxbaiRerankV2

from src.rerankers.abs_reranker import AbsReranker, ScoredChunk

class MixBreadAIReranker(AbsReranker):
    """Uses Cohere reranker models"""
    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
        batch_size: int = 32,
        n_to_rerank : int = 100,
        description : str = ""
        ):
        """Loads a reranker with transformer libray.

        Args:
            model_name (str, optional): HF repo of the mixbread model.
            n_to_rerank (int, optional): amount of chunks to rerank. Defaults to 100.
        """
        super().__init__(n_to_rerank=n_to_rerank, description=description)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = MxbaiRerankV2(model_name)


    def rerank(self, query:str, chunks: list[str]) -> list[ScoredChunk]:
        """
        Args:
            query (str): the query.
            chunks (list[str]): the list of chunks to rerank.

        Returns:
            list[ScoredChunk]: the reranked chunks, ranked by descreasing score.
        """
        results = self.model.rank(
            query, chunks, return_documents=False, sort=False,
            top_k=len(chunks), batch_size=self.batch_size
            )

        return sorted([ScoredChunk(
            text=chunk,
            index=i,
            score=res.score
        ) for i, (chunk, res) in enumerate(zip(chunks, results))],
        key=lambda x: x.score, reverse=True)

