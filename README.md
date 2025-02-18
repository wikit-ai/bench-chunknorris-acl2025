# Parsing and Chunking Benchmark

This repo contains code that can be used to **evaluate the performance of parsing and chunking tools in the context of PDF document ingestion for information retrieval**. It was initially developped in order to estimate the capabilities of [**ChunkNorris**](https://wikit-ai.github.io/chunknorris/) as a serious alternative to commonly used parsing and chunking tools.

It takes into account various metrics such as :
- **energy consumption**: how much energy is used during parsing and chunking
- **parsing speed**: how fast the documents are parsed/chunked during ingestion 
- **recall and NDCG**: how much the parsing/chunking allow to retrieve the relevant chunk regarding queries 

## Getting started

The repo contains various tools :
- **Pipelines**: pipelines are built from various packages meant to be used for parsing and/or chunking. They allow to use these packages with a common interface.
- **Evaluators**: evaluator handle the logic for running a parsing or chunking evaluation, and process the obtained results.

The [examples section](./src/examples/) contains notebook with code snippets to help getting started with these elements.

## Additional information

### Pdf Information Retrieval Evaluation (PIRE) dataset

A **retrieval dataset** has been built especially for the sake of this evaluation. It is based on a series of queries and a corpus of 100 PDF files. The dataset, as well as information about its structure and annotation process can be found [here](https://huggingface.co/datasets/Wikit/PIRE).

### Docker image for parsing evaluation

Parsing is mainly evaluated considering parsing time and energy consumption. **To make sure these measurments are reproducible, it is better to run the parsing exepriments in an isolated environment on a dedicated instance**. The Dockerfile is here to help you get started with building you docker image and run an experiment.
