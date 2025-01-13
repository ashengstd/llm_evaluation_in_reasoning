import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Literal

import litellm
import rich.logging
import rich.progress
from fire import Fire

from llm_evaluation_in_reasoning.data.dataloader import (
    GSM8K,
    BaseBenchDataloader,
    GSMSymbolic,
    SimpleBenchDataloader,
)
from llm_evaluation_in_reasoning.data.question import (
    QUESTION_TYPE_PROMPT_MAP,
    QuestionType,
)
from llm_evaluation_in_reasoning.eval.model import EvalModel, MajorityVoteModel
from llm_evaluation_in_reasoning.eval.scorer import (
    eval_majority_vote,
    eval_question_once,
)

LOGGER_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRTICAL": logging.CRITICAL,
}

RichHander = rich.logging.RichHandler()
ProgressBar = rich.progress.Progress(
    "[progress.description]{task.description}",
    rich.progress.BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    rich.progress.TimeRemainingColumn(),
)


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
    dataset_path: Literal["simple_bench_public.json", "GSM-Symbolic"] = "GSM-Symbolic",
    num_responses: int = 1,
    output_dir: str = "results",
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    logging_level: Literal["INFO", "DEBUG", "ERROR", "WARNING", "CRITICAL"] = "INFO",
    type: str = "main",
    split: Literal["train", "test"] = "test",
    custom_prompt: str | Path | None = None,
):
    """
    Run evaluation benchmark on the specified model and dataset
    with the given parameters
    params:
        model_name: str - name of the model to evaluate
        dataset_path: str - path to the dataset to evaluate on
        num_responses: int - number of responses to collect for majority vote
        output_dir: str - directory to save results
        temp: float - temperature parameter for model
        max_tokens: int - maximum tokens for model
        top_p: float - top p parameter for model
        max_retries: int - maximum retries for model
        system_prompt_path: str - path to system prompt json file
        logging_level: str - logging level
        type: str - type of GSM-Symbolic dataset
        split: str - split of GSM-Symbolic dataset
        custom_prompt: str | Path | None - custom system prompt
    """
    # config log
    if logging_level == "DEBUG":
        litellm.set_verbose = True
    logging.basicConfig(
        level=LOGGER_LEVEL_MAP[logging_level],
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[RichHander],
    )
    # create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # load dataset
    dataset: BaseBenchDataloader
    match dataset_path:
        case "simple_bench_public.json":
            dataset = SimpleBenchDataloader(
                dataset_path,
                progress=ProgressBar,
            )
            pass
        case "GSM-Symbolic":
            dataset = GSMSymbolic(progress=ProgressBar, type=type, split=split)
        case "GSM8K":
            dataset = GSM8K(progress=ProgressBar, type=type, split=split)

    logging.info(f"Loaded {len(dataset)} examples from {dataset_path}")

    # load system prompt
    if custom_prompt is not None:
        if isinstance(custom_prompt, Path):
            custom_prompt = custom_prompt.read_text()
        elif isinstance(custom_prompt, str):
            system_prompt = custom_prompt
        system_prompt = custom_prompt
    else:
        system_prompt = QUESTION_TYPE_PROMPT_MAP[dataset.question_type]
    # initialize eval model and scorer
    model: EvalModel | MajorityVoteModel
    model = EvalModel(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt,
    )
    scorer: Callable[[str | List[str], str | int, QuestionType], bool]
    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)
        scorer = eval_majority_vote
    else:
        scorer = eval_question_once

    # run evaluation
    logging.info(f"Starting evaluation with model: {model_name}")
    results, accuracy = asyncio.run(dataset.evaluate_model(model, scorer))

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


def app():
    Fire(run_benchmark)
