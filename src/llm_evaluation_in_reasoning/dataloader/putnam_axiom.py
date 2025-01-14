from typing import Literal

import rich
import rich.progress
from datasets import DatasetDict, load_dataset

from llm_evaluation_in_reasoning.dataloader.base import BaseBenchDataloader
from llm_evaluation_in_reasoning.util.putnam import convert_answer
from llm_evaluation_in_reasoning.util.putnam_prompt import generate_prompt


class PutnamAXIOM(BaseBenchDataloader):
    def __init__(
        self,
        split: Literal[
            "full_original_236_10_30_2024",
            "func_original_53_10_30_2024",
            "func_variations_265_11_23_2024",
        ] = "full_original_236_10_30_2024",
        prompt_type: Literal[
            "putnam_prompt_type", "gemma", "mistral", "general"
        ] = "putnam_prompt_type",
        progress: rich.progress.Progress = rich.progress.Progress(),
    ):
        if split not in [
            "full_original_236_10_30_2024",
            "func_original_53_10_30_2024",
            "func_variations_265_11_23_2024",
        ]:
            raise ValueError(f"Invalid split: {split}")
        self.dataset: DatasetDict = load_dataset(
            "Putnam-AXIOM/putnam-axiom-dataset", split=split
        )
        self.progress_bar = progress
        self.question_key = "problem"
        self.answer_key = "solution"
        ans_column = generate_prompt(prompt_name=prompt_type, problems=self.dataset)
        self.dataset = self.dataset.remove_columns([self.question_key]).add_column(
            self.question_key, ans_column
        )
        self.dataset = self.dataset.map(convert_answer)
