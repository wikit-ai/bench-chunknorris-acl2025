"""This script purpose is to build simple chunks from the documents.
These chunks can be used as part of a retrieval process to check
that a question generated during labeling only has its answer in one chunk.
"""

import os
import json
import logging
from uuid import uuid4
from tqdm import tqdm
from chunknorris.parsers import PdfParser

# disable chunknorris' info logs
logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)

parser = PdfParser(use_ocr="never")

PDF_DIR = "./data/pdf"
N_WORDS_PER_CHUNKS = 250
OUTPUT_FILENAME = f"chunks_{N_WORDS_PER_CHUNKS}_wordcount.json"

if os.path.exists(f"./{OUTPUT_FILENAME}"):
    with open(f"./{OUTPUT_FILENAME}", "r", encoding="utf8") as file:
        chunks = json.load(file)
else:
    chunks = {}

# list all pdf files
pdf_filepaths: list[str] = []
for root, dirs, files in os.walk(PDF_DIR):
    for file in files:
        if file.endswith(".pdf"):
            pdf_filepaths.append(os.path.abspath(os.path.join(root, file)))

# Parse files to markdown and split into chunks
for filepath in tqdm(pdf_filepaths):
    filename = os.path.basename(filepath)
    if filename in chunks.keys():
        continue
    chunks[filename] = []

    try:
        _ = parser.parse_file(filepath)
    except Exception as e:
        print(filename)
        raise e

    md_string_per_page = parser.to_markdown(keep_track_of_page=True)
    for page_id, text in md_string_per_page.items():
        text_split_by_words = text.split(" ")
        if len(text_split_by_words) > N_WORDS_PER_CHUNKS:
            n_splits = len(text_split_by_words) // N_WORDS_PER_CHUNKS + 1
            n_words_per_chunk = len(text_split_by_words) // n_splits + 1
            splitted_text = [
                " ".join(
                    text_split_by_words[
                        i * n_words_per_chunk : (i + 1) * n_words_per_chunk
                    ]
                )
                for i in range(n_splits)
            ]
        else:
            splitted_text = [text]

        chunks[filename].extend(
            [
                {"page": page_id, "text": subtext, "uuid": str(uuid4()), "order": i}
                for i, subtext in enumerate(splitted_text)
            ]
        )

    with open(OUTPUT_FILENAME, "w", encoding="utf8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)
