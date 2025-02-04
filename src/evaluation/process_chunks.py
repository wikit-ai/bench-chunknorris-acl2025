import json
import os
import datasets
from tqdm import tqdm

import tiktoken
from chromacache import ChromaCache
from chromacache.embedding_functions import SentenceTransformerEmbeddingFunction
from model2vec import StaticModel
from codecarbon import track_emissions

from src.components import Chunk


CHUNKS_DIR = "./results/chunks_data/chunks"

CC = ChromaCache(
    SentenceTransformerEmbeddingFunction(
        "Snowflake/snowflake-arctic-embed-m-v2.0"
    ),
    save_embbedings=True,
    path_to_chromadb="./ChromaDB",
    batch_size=64,
    )

MINISH = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")

TOKENIZER = tiktoken.get_encoding("cl100k_base")

CURRENT_PROJ_NAME = ""

@track_emissions(project_name=CURRENT_PROJ_NAME, offline=True, country_iso_code="FRA")
def embed(chunks_lst: list[str]):
    return CC.encode(chunks_lst)


dataset_dict = datasets.DatasetDict()
for chunks_file in tqdm(os.listdir(CHUNKS_DIR)):
    if not chunks_file == "openparsegpu_chunks.json":
        continue
    with open(os.path.join(CHUNKS_DIR, chunks_file), encoding="utf8") as file:
        all_chunks = json.load(file)
    for chunker_name, chunks in all_chunks.items():
        dataset = datasets.Dataset.from_list(chunks)
        CURRENT_PROJ_NAME = chunks_file.replace("chunks.json", "") + chunker_name
        print(CURRENT_PROJ_NAME)
        dataset = dataset.map(lambda x: {"token_count": len(TOKENIZER.encode(x["text"]))})
        dataset = dataset.add_column("emb_sf_m_v2", embed(dataset["text"]))
        dataset = dataset.add_column("emb_potion_r32M", MINISH.encode(dataset["text"]).tolist())
        # dataset = dataset.add_column("emb_sf_m_v2", CC.encode(dataset["text"]))
        dataset_dict[CURRENT_PROJ_NAME] = dataset
