# Introduction

A project for the evaluation of reasoning in the LLM.

## Install the package

```shell
pip install llm_evaluation_in_reasoning
```

## Run Instructions

Before the eval, place the `simple_bench_public.json` and `system_prompt` in your work dir.
Run benchmark:

```shell
llm_eval --model_name=gpt-4o --dataset_path=simple_bench_public.json
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
