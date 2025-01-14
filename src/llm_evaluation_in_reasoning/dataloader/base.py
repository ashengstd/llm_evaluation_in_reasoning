import logging
from abc import ABC, abstractmethod
from typing import List

import rich
import rich.progress
from datasets import Dataset, DatasetDict

from llm_evaluation_in_reasoning.eval.model import LiteLLM_Model, MajorityVoteModel


class BaseBenchDataloader(ABC):
    dataset: Dataset | DatasetDict

    def __init__(self) -> None:
        self.progress_bar: rich.progress.Progress
        self.question_key: str
        self.answer_key: str

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def inital_default_prompt() -> str:
        return ""

    def process_question(self, example: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, output: str) -> str | int:
        raise NotImplementedError

    @staticmethod
    def eval_single_question(predicted_answer: str | int, answer: str | int) -> bool:
        raise NotImplementedError

    @staticmethod
    def vote_majority(output: List[str | int], answer: str | int) -> bool:
        raise NotImplementedError

    async def evaluate_model(
        self,
        model: LiteLLM_Model | MajorityVoteModel,
        system_prompt: str | None = None,
    ) -> tuple[List[dict], float]:
        results: List[dict] = []
        total_correct = 0
        if not system_prompt:
            logging.info("Evaluating model without default prompt")
            system_prompt = self.inital_default_prompt()
        with self.progress_bar as progress:
            task = progress.add_task("Evaluating model", total=len(self.dataset))
            for i, example in enumerate(self.dataset):
                logging.debug(f"Processing example {i}")
                try:
                    if system_prompt:
                        response = await model.predict(
                            system_prompt + example[self.question_key]
                        )
                    else:
                        response = await model.predict(example[self.question_key])
                    if isinstance(response, list):
                        predicted_answer_list = [
                            self.extract_answer(response_item)
                            for response_item in response
                        ]
                    else:
                        predicted_answer = self.extract_answer(response)
                    if isinstance(response, list):
                        is_correct = self.vote_majority(
                            predicted_answer_list, example[self.answer_key]
                        )
                    else:
                        is_correct = self.eval_single_question(
                            predicted_answer, example[self.answer_key]
                        )
                    results.append(
                        {
                            self.question_key: example[self.question_key],
                            "response": response,
                            self.answer_key: example[self.answer_key],
                            "is_correct": is_correct,
                        }
                    )
                    if is_correct:
                        total_correct += 1
                    progress.update(task, advance=1)
                    logging.info(
                        f"Progress: {i + 1}/{len(self.dataset)} - Accuracy: {total_correct / (i + 1):.2%}"
                    )

                except Exception as e:
                    logging.error(f"Error processing example {i}: {str(e)}")
                    results.append(
                        {self.question_key: example[self.question_key], "error": str(e)}
                    )
            accuracy = total_correct / len(self.dataset)
            return results, accuracy
