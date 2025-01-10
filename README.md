# Introduction

A project for the evaluation of reasoning in the LLM.

## Run Instructions

Run benchmark:

```
python run.py --model_name=gpt-4o --dataset_path=simple_bench_public.json
```

## Setup Instructions

Clone the github repo and cd into it.

```
git clone https://github.com/ashengstd/llm_evaluation_in_reasoning.git
cd llm_evaluation_in_reasoning
```

### Install uv:

The best way to install dependencies is to use `uv`.
If you don't have it installed in your environment, you can install it with the following:

```
curl -LsSf https://astral.sh/uv/install.sh | sh # macOS and Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

### Sync the dependencies

```
uv sync
```

### Create the `.env` file

Create a `.env` file with the following:

```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
...
```
