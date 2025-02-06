import json
import os
import time

from dotenv import load_dotenv, find_dotenv
from ecologits import EcoLogits
from openai import OpenAI
from tqdm import tqdm

from src.components import RagItem
from src.evaluation.metrics.generation.schema import (
    GenerationCollection,
    GenerationItem,
)


load_dotenv(find_dotenv())

EcoLogits.init(providers="openai")


class Generator:
    def __init__(self, path: str):
        """
        Initialize the Generator with the given path.

        Args:
            path (str): The path to the retrieval results file, used to derive the pipeline name.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = "gpt-4o-mini"
        self.name_pipeline = path.split("retrieval_")[1]

    def _get_prompt(self, query: str, context: list[str]) -> list[dict[str, str]]:
        """
        Generate a prompt for the language model based on the query and context.

        Args:
            query (str): The query to be answered.
            context (list[str]): A list of context strings to help answer the query.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the prompt messages.
        """
        context_formatted = "".join(c + "\n\n" for c in context)
        return [
            {
                "role": "system",
                "content": """Your task is to answer the following question helping you with the provided context ('Context').
                You have to use the words from the context to answer the question ('Question'), do not use synonyms.""",
            },
            {"role": "user", "content": f"'Context': : {context_formatted}"},
            {"role": "user", "content": f"## 'Question': {query}"},
        ]

    def _calculate_price(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the price for generating answers based on the number of input and output tokens.

        Args:
            input_tokens (int): The number of input tokens.
            output_tokens (int): The number of output tokens.

        Returns:
            float: The calculated price for the generation.
        """
        price_per_input_token = 0.150 / 10**6  # gpt-4o-mini
        price_per_output_token = 0.600 / 10**6  # gpt-4o-mini

        return (price_per_input_token * input_tokens) + (
            price_per_output_token * output_tokens
        )

    def _build_batches(
        self,
        list_documents: list[str],
        batch_size: int,
    ) -> list[list[str]]:
        """
        Build batches of documents from the given list.

        Args:
            list_documents (list[str]): A list of documents to be batched.
            batch_size (int): The size of each batch.

        Returns:
            list[list[str]]: A list of batches, where each batch is a list of documents.
        """
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_length: int = 0

        for string in list_documents:
            current_batch.append(string)
            current_length += 1
            if current_length >= batch_size:
                batches.append(current_batch)
                current_batch = []
                current_length = 0

        if current_batch:
            batches.append(current_batch)

        return batches

    def generation(self, rag_item: RagItem) -> GenerationCollection:
        """
        Generate an answer for a given RagItem using the LLM.

        Args:
            rag_item (RagItem): The RagItem containing the reference section and top chunks.

        Returns:
            GenerationCollection: A collection containing the reference section and the generated response.
        """
        messages = self._get_prompt(
            query=rag_item.reference_section.query, context=rag_item.top_chunks
        )
        response = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,  # Temperature - Defaults to 1
        )
        generated_item = GenerationItem(
            name_pipeline=self.name_pipeline,
            correct_chunk_index=rag_item.correct_chunk,
            answer=response.choices[0].message.content,
            energy_min=response.impacts.energy.value.min,
            energy_max=response.impacts.energy.value.max,
            gwp_min=response.impacts.gwp.value.min,
            gwp_max=response.impacts.gwp.value.max,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            price=self._calculate_price(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )
        return GenerationCollection(
            reference=rag_item.reference_section,
            generated_response=generated_item,
        )

    def _save_current_batch(self, list_generated: list[GenerationCollection]):
        """
        Save the current batch of generated responses to a JSON file.

        Args:
            list_generated (list[GenerationCollection]): A list of generated responses to save.

        Returns:
            None
        """
        directory = "results_generation"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "results.json")

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                results_history = json.load(f)
        else:
            results_history = []

        list_already_included = [
            reference["reference"]["query"] for reference in results_history
        ]

        for item in list_generated:
            if item.reference.query in list_already_included:
                for reference in results_history:
                    if reference["reference"]["query"] == item.reference.query:
                        if (
                            item.generated_response.model_dump()
                            not in reference["generated_answers"]
                        ):
                            reference["generated_answers"].append(
                                item.generated_response.model_dump()
                            )
                        break
            else:
                new_reference = {
                    "reference": item.reference.model_dump(),
                    "generated_answers": [item.generated_response.model_dump()],
                }
                results_history.append(new_reference)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results_history, f, ensure_ascii=False, indent=4)

    def generation_per_batch(self, batch: int, items_to_generate: list[RagItem]):
        """
        Generate answers in batches for the given list of RagItems.

        Args:
            batch (int): The size of each batch.
            items_to_generate (list[RagItem]): A list of RagItems to generate answers for.

        Returns:
            None
        """
        batched_items = self._build_batches(
            list_documents=items_to_generate, batch_size=batch
        )
        for batch in tqdm(batched_items):
            generated_batch = [self.generation(item) for item in batch]
            self._save_current_batch(list_generated=generated_batch)
            time.sleep(2)
