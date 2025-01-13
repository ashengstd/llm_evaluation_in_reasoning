import json
import logging
import re
from abc import ABC
from functools import partial
from typing import Callable, List, Literal

import rich
import rich.progress
from datasets import Dataset, DatasetDict, load_dataset

from llm_evaluation_in_reasoning.data.question import QuestionType
from llm_evaluation_in_reasoning.eval.model import EvalModel, MajorityVoteModel


class BaseBenchDataloader(ABC):
    dataset: Dataset | DatasetDict

    def __init__(self) -> None:
        self.progress_bar: rich.progress.Progress
        self.question_key: str
        self.answer_key: str
        self.question_type: QuestionType

    def __len__(self) -> int:
        return len(self.dataset)

    async def evaluate_model(
        self,
        model: EvalModel | MajorityVoteModel,
        scorer: Callable[[str | List[str], str | int, QuestionType], bool],
    ) -> tuple[List[dict], float]:
        results: List[dict] = []
        total_correct = 0
        with self.progress_bar as progress:
            task = progress.add_task("Evaluating model", total=len(self.dataset))
            for i, example in enumerate(self.dataset):
                logging.debug(f"Processing example {i}")
                try:
                    response = await model.predict(example[self.question_key])
                    is_correct = scorer(
                        response, example[self.answer_key], self.question_type
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
                        f"Progress: {i+1}/{len(self.dataset)} - Accuracy: {total_correct/(i+1):.2%}"
                    )

                except Exception as e:
                    logging.error(f"Error processing example {i}: {str(e)}")
                    results.append(
                        {self.question_key: example[self.question_key], "error": str(e)}
                    )
            accuracy = total_correct / len(self.dataset)
            return results, accuracy


class SimpleBenchDataloader(BaseBenchDataloader):
    def __init__(self, file_path: str, progress: rich.progress.Progress) -> None:
        with open(file_path, "r") as file:
            data = json.load(file)
        self.dataset = data["eval_data"]
        self.progress_bar = progress
        self.question_key = "prompt"
        self.answer_key = "answer"
        self.question_type = QuestionType.MULTIPLE_CHOICE


class HFDataloader(BaseBenchDataloader, ABC):
    def __init__(
        self,
        path: str,
        name: str,
        split: Literal["test", "train"] = "test",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        self.dataset: DatasetDict = load_dataset(path=path, name=name, split=split)
        self.dataset = self.dataset["test"]
        self.progress_bar = progress
        self.question_key = "question"
        self.question_type = QuestionType.BLANK_FILL
        self.answer_key = "answer"
        extract_with_params = partial(answer2int_gsm, anwser_key=self.answer_key)
        self.dataset = self.dataset.map(extract_with_params)


def answer2int_gsm(example: Dataset, anwser_key: str) -> int:
    answer_text = example[anwser_key]
    match = re.search(r"####\s*(\d+)", answer_text)
    if match:
        example[anwser_key] = int(match.group(1))
    else:
        example[anwser_key] = -1
    return example


class GSMSymbolic(HFDataloader):
    def __init__(
        self,
        type: Literal["main", "p1", "p2"] = "main",
        split: Literal["test", "train"] = "test",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        self.dataset: DatasetDict = load_dataset(
            path="apple/GSM-Symbolic", name=type, split=split
        )
        self.progress_bar = progress
        self.question_key = "question"
        self.question_type = QuestionType.BLANK_FILL
        self.answer_key = "answer"
        extract_with_params: Callable = partial(
            answer2int_gsm, anwser_key=self.answer_key
        )
        self.dataset = self.dataset.map(extract_with_params)


class GSM8K(HFDataloader):
    def __init__(
        self,
        type: Literal["main", "socratic"] = "main",
        split: Literal["test", "train"] = "test",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        self.dataset: DatasetDict = load_dataset(
            path="openai/gsm8k", name=type, split=split
        )
        self.progress_bar = progress
        self.question_key = "question"
        self.question_type = QuestionType.BLANK_FILL
        self.answer_key = "answer"
        extract_with_params = partial(answer2int_gsm, anwser_key=self.answer_key)
        self.dataset = self.dataset.map(extract_with_params)
