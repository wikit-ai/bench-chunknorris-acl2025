import json
import logging
import random
from collections import Counter

import datasets
import torch
from sentence_transformers.util import cos_sim
import streamlit as st

from chromacache import ChromaCache
from chromacache.embedding_functions import SentenceTransformerEmbeddingFunction
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()


# Config variables
HF_ANNOTATION_REPO = (
    "Wikit/retrieval-pdf-acl2025"  # the HF repo used to store annotations
)
CHUNKS_PATH = "./tools/chunks_250_wordcount.json"
MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"  # a HF path to an embedding model
PATH_TO_CHROMDB = "./tools/ChromaDB"
TOPN_TO_RETRIEVE = 10
FILES_PER_USER = "./tools/files_per_annotator.json"


@st.cache_resource
def load_annotations(user_name: str):
    """Loads the 'backend' of the app, or initializes it
    if not existing backend is found

    Returns:
        dict: the annotations (which is a JSON)
    """
    annotations = datasets.load_dataset(HF_ANNOTATION_REPO, user_name)["train"]
    return annotations


@st.cache_resource
def load_chunks(chunks_path: str):
    """Initializes the backend file.
    Uses the JSON file containing the chunks
    output by the get_standard_chunks.py script

    Args:
        chunks_path (str): the path the the json file containing the chunks
    """
    with open(chunks_path, "r", encoding="utf8") as file:
        chunks = json.load(file)

    chunks = [
        chunk | {"source_file": filename}
        for filename, filechunks in chunks.items()
        for chunk in filechunks
    ]

    dataset = datasets.Dataset.from_list(chunks)
    dataset = dataset.add_column("embedding", MODEL.encode(dataset["text"]))

    return dataset


def get_new_filename():
    """Gets a new filename among all files for which
    we don't have annotations for.
    """
    annotated_filenames = Counter(st.session_state.annotations["source_file"])
    annotated_filenames = [
        filename for filename, count in annotated_filenames.items() if count >= 3
    ]

    # if file has already 3 annotations, get new file
    if (
        st.session_state.current_filename in annotated_filenames
        or not st.session_state.current_filename
    ):
        non_annotated_files = [
            filename
            for filename in st.session_state.user_files_to_annotate
            if filename not in annotated_filenames
        ]

        return random.choice(non_annotated_files)

    return st.session_state.current_filename


def save_backend():
    """Saves the annotations to a HF repo"""
    filename = f"{st.session_state.user_name.lower()}.json"
    st.session_state.annotations.to_json(
        filename,
        lines=False,
        force_ascii=False,
        indent=4,
    )

    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"./{filename}",
        path_in_repo=filename,
        repo_id=HF_ANNOTATION_REPO,
        repo_type="dataset",
    )


@st.cache_resource
def setup_chroma_embedder(model_name: str, path_to_chromadb: str):
    """Sets up a chromaDB embedder to retrieve top documents

    Args:
        model_name (str): an OpenAI model name
        path_to_chromadb (str): the path to the chromaDB storage

    Returns:
        ChromaDBEmbedder: the embedder
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Embedding chunks using HF model %s on %s", model_name, device)
    return ChromaCache(
        SentenceTransformerEmbeddingFunction(model_name),
        path_to_chromadb=path_to_chromadb,
        save_embbedings=True,
        batch_size=16,
    )


def get_files_for_user(user_name: str):
    """Considering a provided user name, gets the files that they should label"""
    user_name = user_name.lower()

    with open(FILES_PER_USER, "r", encoding="utf8") as file:
        files_per_user = json.load(file)

    if user_name not in files_per_user:
        st.warning(
            "Invalid user name. Available user names are list(files_per_user.keys())",
            icon="⚠️",
        )
        return

    st.session_state.user_files_to_annotate = [
        file["pdf_file_name"] for file in files_per_user[user_name]
    ]


def retrieve_topn_docs(query: str):
    """Retrieve the top documents corresponding to the user's input query,
    so that related chunks can be added.

    Updates the uuids of the chunks to display

    Args:
        query (str): the user's query
    """
    LOGGER.info("Retrieving chunks")
    query_emb = torch.tensor(MODEL.encode(query), dtype=torch.float)
    cosims = cos_sim(query_emb, st.session_state.chunks_dataset["embedding"]).squeeze()
    top_indexes = torch.topk(cosims, k=TOPN_TO_RETRIEVE)
    st.session_state.top_chunks = [
        st.session_state.chunks_dataset[idx] for idx in top_indexes.indices.tolist()
    ]


def save_and_load_new_page():
    """Save the current backend state, reset user's input and display new chunk to label"""
    if not st.session_state.anno_page.isnumeric():
        st.warning("Annotated page should be an int", icon="⚠️")
        return

    st.session_state.annotations = load_annotations(st.session_state.user_name)

    st.session_state.annotations = datasets.concatenate_datasets(
        [
            st.session_state.annotations,
            datasets.Dataset.from_list(
                [
                    {
                        "source_file": st.session_state.current_filename,
                        "query": st.session_state.query,
                        "target_page": int(st.session_state.anno_page),
                        "target_passage": st.session_state.anno_passage,
                    }
                ]
            ),
        ]
    )

    save_backend()
    st.session_state.current_filename = get_new_filename()

    st.session_state.query = ""
    st.session_state.anno_page = ""
    st.session_state.anno_passage = ""
    st.session_state.top_chunks = []

    annotated_docs = len(st.session_state.annotations)
    st.success(f"Label saved. (Current labeled docs : {annotated_docs})", icon="✅")


# Setup logger
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

# Setup backend
### Initialize session state
MODEL = setup_chroma_embedder(MODEL_NAME, PATH_TO_CHROMDB)


if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "chunks_dataset" not in st.session_state:
    st.session_state.chunks_dataset = load_chunks(CHUNKS_PATH)

if "current_filename" not in st.session_state:
    st.session_state.current_filename = ""

if "top_chunks" not in st.session_state:
    st.session_state.top_chunks = []

if "user_files_to_annotate" not in st.session_state:
    st.session_state.user_files_to_annotate = []

# Setup user
st.text_input("What is your name ?", key="user_name")

if st.session_state.user_name:
    get_files_for_user(st.session_state.user_name)
if not st.session_state.annotations:
    st.session_state.annotations = load_annotations(st.session_state.user_name)

if st.session_state.user_files_to_annotate:
    ### Choose and display random chunk
    st.session_state.current_filename = get_new_filename()

    st.text("Source file : " + st.session_state.current_filename)
    st.text_input("Query ", key="query")
    st.text_input("Page containing the answer", key="anno_page")
    st.text_input("Target passage, as short as possible", key="anno_passage")

    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "Retrieve related chunks",
            on_click=retrieve_topn_docs,
            args=(st.session_state.query,),
            type="primary",
            disabled=False,
            use_container_width=True,
        )
    with col2:
        st.button(
            "Skip",
            on_click=get_new_filename(),
            type="secondary",
            disabled=False,
            use_container_width=False,
        )

    if st.session_state.top_chunks:
        st.subheader("Chunks likely to contain answer", divider="blue")
        for i, chunk in enumerate(st.session_state.top_chunks):
            cont = st.container(border=True)
            with cont:
                st.markdown(
                    "Source file : "
                    + chunk["source_file"]
                    + " page "
                    + str(chunk["page"])
                )
                st.markdown(chunk["text"])
        st.button(
            "Submit",
            on_click=save_and_load_new_page,
        )
