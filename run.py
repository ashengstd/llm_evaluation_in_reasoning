import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from fire import Fire  # type: ignore

from utils.models import LiteLLMModel, MajorityVoteModel
from utils.scorers import eval_majority_vote, eval_multi_choice

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataset(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["eval_data"]


def load_system_prompt(prompt_path: str, prompt_name: str = "multiple_choices") -> str:
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if prompt_name not in data["prompts"]:
                raise KeyError(f"prompt name '{prompt_name}' invalid")
            return data["prompts"][prompt_name]
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found: {prompt_path}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON error: {prompt_path}")


async def evaluate_model(model, dataset: List[Dict], scorer):
    results = []
    total_correct = 0

    for i, example in enumerate(dataset):
        try:
            response = await model.predict(example["prompt"])
            is_correct = scorer(response, example["answer"])
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
                f"Progress: {i+1}/{len(dataset)} - Accuracy: {total_correct/(i+1):.2%}"
            )

        except Exception as e:
            logging.error(f"Error processing example {i}: {str(e)}")
            results.append({"prompt": example["prompt"], "error": str(e)})

    accuracy = total_correct / len(dataset)
    return results, accuracy


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
):
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 加载数据集
    dataset = load_dataset(dataset_path)
    logging.info(f"Loaded {len(dataset)} examples from {dataset_path}")

    # 加载系统提示词
    system_prompt = load_system_prompt(system_prompt_path, "multiple_choices")

    # 初始化模型]
    model: LiteLLMModel | MajorityVoteModel
    model = LiteLLMModel(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt,
    )
    scorer: Callable[[str, str], Any] | Callable[[List[str], str], Any]
    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)
        scorer = eval_majority_vote
    else:
        scorer = eval_multi_choice

    # 运行评估
    logging.info(f"Starting evaluation with model: {model_name}")
    results, accuracy = asyncio.run(evaluate_model(model, dataset, scorer))

    # 保存结果
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
