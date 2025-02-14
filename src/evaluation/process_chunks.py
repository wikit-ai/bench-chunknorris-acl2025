import json
import os
import datasets
from tqdm import tqdm

import tiktoken
from chromacache import ChromaCache
from chromacache.embedding_functions import SentenceTransformerEmbeddingFunction
from model2vec import StaticModel
from codecarbon import track_emissions


class ChunksProcessor:
    def __init__(self, hf_model_repo: str = "Snowflake/snowflake-arctic-embed-m-v2.0", batch_size :int = 64):
        self._hf_model_repo = hf_model_repo
        self.batch_size = batch_size
        self.cc = ChromaCache(
            SentenceTransformerEmbeddingFunction(hf_model_repo),
            save_embbedings=True,
            path_to_chromadb="./ChromaDB",
            batch_size=self.batch_size,
            )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.minish = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")

    @property
    def hf_model_repo(self) -> str:
        """The HF model repo to use for embedding the chunks"""
        return self._hf_model_repo
    
    @hf_model_repo.setter
    def hf_model_repo(self, value):
        self._hf_model_repo = value
        self.cc = ChromaCache(
            SentenceTransformerEmbeddingFunction(self.hf_model_repo),
            save_embbedings=True,
            path_to_chromadb="./ChromaDB",
            batch_size=self.batch_size,
            )

    def process_chunks(
        self,
        chunks_folder : str = "./results/chunks_data/chunks", track_emissions: bool = False
        ) -> datasets.DatasetDict:
        """Processes the chunks files obtained from the an evaluator.get_chunks()
        - Counts tokens of each chunk
        - Embeds each chunk with Potion Retrieval 32M and provided HF model

        NOTE : chunks files must be json files of chunks. They should be called "<something>_chunks.json"

        Args:
            chunks_folder (str, optional): the folder where chunks are stored. Defaults to "./results/chunks_data/chunks".
            track_emissions (bool, optional): whether emissions due to embedding should be tracker. Defaults to False.

        Returns:
            datasets.DatasetDict: a dataset dict where each combination of parser/chunker us a split.
        """
        dataset_dict = datasets.DatasetDict()
        for chunks_file in tqdm(os.listdir(chunks_folder)):
            with open(os.path.join(chunks_folder, chunks_file), encoding="utf8") as file:
                all_chunks = json.load(file)
            for chunker_name, chunks in all_chunks.items():
                dataset = datasets.Dataset.from_list(chunks)
                split_name = chunks_file.replace("chunks.json", "_") + chunker_name
                print(split_name)
                dataset = dataset.map(lambda x: {"token_count": len(self.tokenizer.encode(x["text"]))})
                dataset = dataset.add_column("emb_potion_r32M", self.minish.encode(dataset["text"]).tolist())
                if track_emissions:
                    dataset = dataset.add_column("emb_sf_m_v2", self.embed(dataset["text"]))
                else:
                    dataset = dataset.add_column("emb_sf_m_v2", self.cc.encode(dataset["text"]))
                dataset_dict[split_name] = dataset

        return dataset_dict

    @track_emissions(project_name="", offline=True, country_iso_code="FRA")
    def embed(self, chunks_lst: list[str]):
        return self.cc.encode(chunks_lst)
