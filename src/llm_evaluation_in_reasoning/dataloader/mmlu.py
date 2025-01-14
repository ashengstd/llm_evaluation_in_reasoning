import logging
import re
from typing import Literal

import rich
import rich.progress
from datasets import DatasetDict, load_dataset

from llm_evaluation_in_reasoning.dataloader.base import BaseBenchDataloader


class CaisMMLU(BaseBenchDataloader):
    def __init__(
        self,
        type="all",
        split: Literal["test", "train"] = "test",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        self.dataset: DatasetDict = load_dataset("cais/mmlu", split=split, name=type)
        self.progress_bar = progress
        self.question_key = "question"
        self.answer_key = "answer"
        self.choices_key = "choices"
        self.dataset = self.dataset.map(self.process_question)

    @staticmethod
    def process_question(example: dict) -> dict:
        question = example["question"]
        choices = example["choices"]
        example["question"] = (
            f"{question} The choices below are {choices}, which one is correct?"
        )
        return example

    @staticmethod
    def inital_default_prompt():
        return "You are tasked with answering questions from a variety of domains. Please provide your answers as a numbered list in the following format:\n\n1. [First answer]\n2. [Second answer]\n3. [Third answer]\n4. [Fourth answer]\n5. [Fifth answer] .....\nBelow is the question, please provide your answers using the required format: Final answer: x (x is the number of the correct answer)"

    @staticmethod
    def eval_single_question(predicted_answer: int, answer: int) -> bool:
        logging.info(f"Predicted answer: {predicted_answer}, Ground truth: {answer}")
        return predicted_answer == int(answer)

    @staticmethod
    def vote_majority(output: list[str], answer: str) -> bool:
        max_vote = max(set(output), key=output.count)
        logging.info(f"Output: {max_vote}, Ground truth: {answer}")
        return output.count(answer) > len(output) / 2

    @staticmethod
    def extract_answer(output: str) -> int:
        try:
            output = output.replace(",", " ")
            match = re.findall(r"-?\d+\.?\d*", output)[-1]
            answer = int((str(match)))
            logging.info(f"Answer extracted: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error extracting answer: {e}")
            return -1
