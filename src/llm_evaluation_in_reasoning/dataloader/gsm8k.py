import logging
import re
from functools import partial
from typing import Literal

import rich.progress
from datasets import DatasetDict, load_dataset

from llm_evaluation_in_reasoning.dataloader.base import BaseBenchDataloader


class GSM8K(BaseBenchDataloader):
    def __init__(
        self,
        type: Literal["main", "socratic"] = "main",
        split: Literal["test", "train"] = "test",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        if type not in ["main", "socratic"]:
            logging.error(f"Invalid type: {type}, must be one of ['main', 'socratic']")
            raise ValueError(
                f"Invalid type: {type}, must be one of ['main', 'socratic']"
            )
        if split not in ["test", "train"]:
            logging.error(f"Invalid split: {split}, must be one of ['test', 'train']")
            raise ValueError(
                f"Invalid split: {split}, must be one of ['test', 'train']"
            )
        self.dataset: DatasetDict = load_dataset(
            path="openai/gsm8k", name=type, split=split
        )
        self.progress_bar = progress
        self.question_key = "question"
        self.answer_key = "answer"
        extract_with_params = partial(self.answer2int_gsm, anwser_key=self.answer_key)
        self.dataset = self.dataset.map(extract_with_params)

    @staticmethod
    def answer2int_gsm(example: dict, anwser_key: str) -> dict:
        answer_text = example[anwser_key]
        match = re.search(r"####\s*(\d+)", answer_text)
        if match:
            example[anwser_key] = int(match.group(1))
        else:
            example[anwser_key] = -1
        return example

    @staticmethod
    def extract_answer(output: str) -> int:
        try:
            output = output.replace(",", " ")
            match = re.findall(r"-?\d+\.?\d*", output)[-1]
            answer = int(float(str(match)))
            logging.info(f"Answer extracted: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error extracting answer: {e}")
            return -1

    @staticmethod
    def eval_single_question(predicted_answer: int, answer: int) -> bool:
        logging.info(f"Predicted answer: {predicted_answer}, Ground truth: {answer}")
        return predicted_answer == answer

    @staticmethod
    def vote_majority(output: list[int], answer: int) -> bool:
        max_vote = max(set(output), key=output.count)
        logging.info(f"Output: {max_vote}, Ground truth: {answer}")
        return output.count(answer) > len(output) / 2

    @staticmethod
    def inital_default_prompt() -> str:
        return "You are a creative and intuitive reasoning expert who excels at solving abstract problems. For each question:\n\n1. Trust your instincts and initial impressions\n2. Consider the problem as a whole\n3. Think outside the box and explore unconventional ideas\n4. Use your creativity to generate innovative solutions\n5. Follow your intuition to reach a unique and insightful conclusion\n\nPresent your answer in the following format:\n\nFinal Answer: X\n\nwhere X is a numerical value or a word that best completes the sentence."
