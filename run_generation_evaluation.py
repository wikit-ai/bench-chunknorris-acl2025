from src.evaluation.pipeline_evaluator import PipelinePerformanceEvaluator


pipeline_evaluator = PipelinePerformanceEvaluator()

pipeline_evaluator.run_generation(
    file_path=r"results_retrieval\emb_sf_m_v2\retrieval_default_docling.json",
    batch_size=15,
)
