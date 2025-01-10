import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List

from dotenv import load_dotenv
from fire import Fire

from data.dataloader import BaseBenchDataset, SimpleBenchDataset
from data.question import QuestionType
from eval.model import LiteLLMModel, MajorityVoteModel
from eval.scorer import eval_majority_vote, eval_single_question

load_dotenv()


class LOGGER_LEVEL(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def load_system_prompt(
    prompt_path: str, question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
) -> str:
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            prompt_name = question_type.value
            logging.info(f"Loaded system prompt for {prompt_name}")
            if prompt_name not in data["prompts"]:
                logging.error(f"prompt name '{prompt_name}' invalid")
                raise KeyError(f"prompt name '{prompt_name}' invalid")
            return data["prompts"][prompt_name]
    except FileNotFoundError:
        logging.error(f"file not found: {prompt_path}")
        raise FileNotFoundError(f"file not found: {prompt_path}")
    except json.JSONDecodeError:
        logging.error(f"JSON error: {prompt_path}")
        raise ValueError(f"JSON error: {prompt_path}")


def run_benchmark(
    model_name: str = "op-qwen-2.5-0.5b",
    dataset_path: str = "simple_bench_public.json",
    num_responses: int = 1,
    output_dir: str = "results",
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    system_prompt_path: str = "system_prompt.json",
    logging_level: LOGGER_LEVEL = LOGGER_LEVEL.INFO,
):
    # config log
    logging.basicConfig(
        level=logging_level.value, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # load dataset
    dataset: BaseBenchDataset = SimpleBenchDataset(dataset_path)
    logging.info(f"Loaded {len(dataset)} examples from {dataset_path}")

    # load system prompt
    system_prompt = load_system_prompt(system_prompt_path, QuestionType.MULTIPLE_CHOICE)

    # initialize model and scorer
    model: LiteLLMModel | MajorityVoteModel
    model = LiteLLMModel(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt,
    )
    scorer: (
        Callable[[str, str, QuestionType], Any]
        | Callable[[List[str], str, QuestionType], Any]
    )
    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)
        scorer = eval_majority_vote
    else:
        scorer = eval_single_question

    # run evaluation
    logging.info(f"Starting evaluation with model: {model_name}")
    results, accuracy = asyncio.run(
        dataset.evaluate_model(
            model, scorer, question_type=QuestionType.MULTIPLE_CHOICE
        )
    )

    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = Path(output_dir) / f"results_{model_name}_{timestamp}.json"

    output = {
        "model_name": model_name,
        "accuracy": accuracy,
        "num_responses": num_responses,
        "parameters": {"temperature": temp, "max_tokens": max_tokens, "top_p": top_p},
        "results": results,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    logging.info(f"Evaluation complete - Final accuracy: {accuracy:.2%}")
    logging.info(f"Results saved to: {result_file}")


if __name__ == "__main__":
    Fire(run_benchmark)
