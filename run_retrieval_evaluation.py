from dotenv import load_dotenv, find_dotenv

from src.evaluation.pipeline_evaluator import PipelinePerformanceEvaluator

load_dotenv(find_dotenv())

pipeline_evaluator = PipelinePerformanceEvaluator()

dict_pipe_chunker = {
    "base": ["PageChunker", "RecursiveCharacterChunker"],
    "chunknorris": ["Default", "PageChunker", "RecursiveCharacterChunker"],
    "docling": ["Default", "PageChunker", "RecursiveCharacterChunker"],
    "marker": ["PageChunker", "RecursiveCharacterChunker"],
    "openparsecpu": ["Default", "PageChunker", "RecursiveCharacterChunker"],
    "openparsegpu": ["Default", "PageChunker", "RecursiveCharacterChunker"],
}

for ingest_pipeline in list(dict_pipe_chunker.keys()):
    print(ingest_pipeline.upper())
    for chunker in dict_pipe_chunker[ingest_pipeline]:
        print("-----> ", chunker)
        pipeline_evaluator.evaluate_retrieval(
            top_k=10, chunker=chunker, pipeline=ingest_pipeline
        )
