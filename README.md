# Introduction

A project for the evaluation of reasoning in the LLM.

## Install the package

```shell
pip install llm_evaluation_in_reasoning
```

## Run Instructions

Before the eval, place the `simple_bench_public.json` in your work dir.
Support also `GSM-Symbolic`, `GSM8K`
Run benchmark:

```shell
llm_eval --model_name=op-qwen-2.5-0.5b --dataset_path=simple_bench_public.json # run llm_eval --help to see help informations
```

## Model support

```py
MODEL_MAP = {
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4-turbo": "gpt-4-turbo",
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini",
    "claude-3-opus-20240229": "claude-3-opus-20240229",
    "command-r-plus": "command-r-plus-08-2024",
    "gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash-exp",
    "llama3-405b-instruct": "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-1.5-pro-002": "gemini/gemini-1.5-pro-002",
    "mistral-large": "mistral/mistral-large-2407",
    "grok-2": "openrouter/x-ai/grok-2",
    "op-qwen-2.5-0.5b": "ollama/qwen2.5:0.5b",
    "op-qwen-2.5-7b": "ollama/qwen2.5:7b",
    "op-deepseek-v3": "openrouter/deepseek/deepseek-chat",
    "op-gemini-2.0-flash-free": "openrouter/google/gemini-2.0-flash-thinking-exp:free",
    "op-o1-preview": "openrouter/openai/o1-preview",
}
```

# Build the project

## Setup Instructions

Clone the github repo and cd into it.

```shell
git clone https://github.com/ashengstd/llm_evaluation_in_reasoning.git
cd llm_evaluation_in_reasoning
```

### Install uv:

The best way to install dependencies is to use `uv`.
If you don't have it installed in your environment, you can install it with the following:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh # macOS and Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

### Sync the dependencies

```shell
uv sync
```

### Create the `.env` file

Create a `.env` file with the following:

```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
...
```
