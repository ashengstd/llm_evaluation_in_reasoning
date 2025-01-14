import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import litellm
import rich.logging
import rich.progress
from fire import Fire

from llm_evaluation_in_reasoning.dataloader import (
    GSM8K,
    BaseBenchDataloader,
    CaisMMLU,
    GSMSymbolic,
    SimpleBench,
)
from llm_evaluation_in_reasoning.eval.model import LiteLLM_Model, MajorityVoteModel

LOGGER_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

RichHander = rich.logging.RichHandler()
ProgressBar = rich.progress.Progress(
    "[progress.description]{task.description}",
    rich.progress.BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    rich.progress.TimeRemainingColumn(),
)


def run_benchmark(
    model_name: str = "op-qwen-2.5-0.5b",
    dataset: Literal["SimpleBench", "GSM-Symbolic", "GSM8K", "MMLU"] = "GSM-Symbolic",
    num_responses: int = 1,
    output_dir: str = "results",
    temp: float = 0.9,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    logging_level: Literal["INFO", "DEBUG", "ERROR", "WARNING", "CRITICAL"] = "INFO",
    type: str = "main",
    split: Literal["train", "test"] = "test",
    # putnam_prompt_type: Literal[
    #     "", "general_few_shot_prompt"
    # ] = "general_few_shot_prompt",
    custom_prompt: str | Path | None = None,
):
    """
    Run evaluation benchmark on the specified model and dataset
    with the given parameters
    params:
        model_name: str - name of the model to evaluate
        dataset: str - name to the dataset to evaluate on
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

    # load dataloader
    dataloader: BaseBenchDataloader
    match dataset:
        case "SimpleBench":
            dataloader = SimpleBench(
                progress=ProgressBar,
            )
        case "GSM-Symbolic":
            dataloader = GSMSymbolic(progress=ProgressBar, type=type, split=split)
        case "GSM8K":
            dataloader = GSM8K(progress=ProgressBar, type=type, split=split)
        # case "Putnam-AXIOM":
        #     dataloader = PutnamAXIOM(
        #         progress=ProgressBar, split=split, prompt_type=putnam_prompt_type
        #     )
        case "MMLU":
            dataloader = CaisMMLU(progress=ProgressBar, split=split)
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    logging.info(f"Loaded {len(dataloader)} examples from {dataset}")
    system_prompt: str | None = ""
    # load system prompt
    if custom_prompt is not None:
        if isinstance(custom_prompt, Path):
            custom_prompt = custom_prompt.read_text()
        elif isinstance(custom_prompt, str):
            system_prompt = custom_prompt
        system_prompt = custom_prompt

    # initialize eval model and scorer
    model: LiteLLM_Model | MajorityVoteModel
    model = LiteLLM_Model(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt,
    )
    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)
    elif num_responses < 1:
        raise ValueError("num_responses must be greater than 1 and an integer")

    # run evaluation
    logging.info(f"Starting evaluation with model: {model_name}")
    results, accuracy = asyncio.run(dataloader.evaluate_model(model))

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


if __name__ == "__main__":
    app()
