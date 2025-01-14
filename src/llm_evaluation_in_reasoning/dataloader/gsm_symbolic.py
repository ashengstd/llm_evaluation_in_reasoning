import logging
from functools import partial
from typing import Callable, Literal

import rich
import rich.progress
from datasets import DatasetDict, load_dataset

from llm_evaluation_in_reasoning.dataloader.gsm8k import GSM8K


class GSMSymbolic(GSM8K):
    def __init__(
        self,
        type: Literal["main", "p1", "p2"] = "main",
        split: Literal["test", "train"] = "test",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        if type not in ["main", "p1", "p2"]:
            logging.error(f"Invalid type: {type}")
            raise ValueError(f"Invalid type: {type}")
        if split not in ["test", "train"]:
            logging.error(f"Invalid split: {split}")
            raise ValueError(f"Invalid split: {split}")
        self.dataset: DatasetDict = load_dataset(
            path="apple/GSM-Symbolic", name=type, split=split
        )
        self.progress_bar = progress
        self.question_key = "question"
        self.answer_key = "answer"
        extract_with_params: Callable = partial(
            self.answer2int_gsm, anwser_key=self.answer_key
        )
        self.dataset = self.dataset.map(extract_with_params)
