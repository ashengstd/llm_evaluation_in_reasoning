import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

from datasets import Dataset, load_dataset

from data.question import QuestionType
from eval.model import LiteLLMModel, MajorityVoteModel


class BaseBenchDataset(ABC):
    dataset: Dataset

    @abstractmethod
    def __init__(self, *arg) -> None:
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    @abstractmethod
    async def evaluate_model(self, model, scorer, question_type) -> Any:
        pass


class SimpleBenchDataset(BaseBenchDataset):
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            data = json.load(file)
            self.dataset = data["eval_data"]

    async def evaluate_model(
        self,
        model: LiteLLMModel | MajorityVoteModel,
        scorer: Callable[[str | list[str], str, QuestionType], bool],
        question_type: QuestionType,
    ) -> tuple[list[dict], float]:
        results: list[dict] = []
        total_correct = 0

        for i, example in enumerate(self.dataset):
            try:
                response = await model.predict(example["prompt"])
                is_correct = scorer(response, example["answer"], question_type)
                results.append(
                    {
                        "prompt": example["prompt"],
                        "response": response,
                        "correct": example["answer"],
                        "is_correct": is_correct,
                    }
                )
                if is_correct:
                    total_correct += 1

                logging.info(
                    f"Progress: {i+1}/{len(self.dataset)} - Accuracy: {total_correct/(i+1):.2%}"
                )

            except Exception as e:
                logging.error(f"Error processing example {i}: {str(e)}")
                results.append({"prompt": example["prompt"], "error": str(e)})

        accuracy = total_correct / len(self.dataset)
        return results, accuracy


class GSMSymbolic(BaseBenchDataset):
    def __init__(self, type: Literal["main", "p1", "p2"] = "main"):
        self.dataset = load_dataset("apple/GSM-Symbolic", type)
