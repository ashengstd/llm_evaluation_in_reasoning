# llm_evaluation_in_reasoning

[![example workflow](https://github.com/ashengstd/llm_evaluation_in_reasoning/actions/workflows/publish-pypi-release.yml/badge.svg)](https://github.com/ashengstd/llm_evaluation_in_reasoning) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm_evaluation_in_reasoning)](https://pypi.org/project/llm_evaluation_in_reasoning) [![PyPI](https://img.shields.io/pypi/v/llm_evaluation_in_reasoning.svg)](https://pypi.org/project/llm_evaluation_in_reasoning/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/llm_evaluation_in_reasoning.svg)](https://pypi.org/project/llm_evaluation_in_reasoning/) [![GitHub License](https://img.shields.io/github/license/ashengstd/llm_evaluation_in_reasoning)](https://github.com/ashengstd/llm_evaluation_in_reasoning) [![GitHub Release](https://img.shields.io/github/v/release/ashengstd/llm_evaluation_in_reasoning)](https://github.com/ashengstd/llm_evaluation_in_reasoning)

A project for evaluating reasoning capabilities in large language models (LLMs).

**Read this in other languages: [English](https://github.com/ashengstd/llm_evaluation_in_reasoning/blob/main/README.md), [中文](https://github.com/ashengstd/llm_evaluation_in_reasoning/blob/main/README_zh.md).**

## Run the eval

### Install the package

```shell
pip install llm_evaluation_in_reasoning
```

### Create the `.env` file

Create a `.env` file with the following:

```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
...
```

The api key you provided will be used to fetch the valid models supported by `Litellm`.

### Run Instructions

Support `GSM-Symbolic`, `GSM8K`, `MMLU`, `SimpleBench`
To run a benchmark:

```shell
llm_eval --model_name=ollama/qwen2.5:0.5b --dataset=SimpleBench # run llm_eval --help to see help information
```

## Model support

Model support is based on `Litellm`, see the docs here [Litellm Providers](https://docs.litellm.ai/docs/providers)

## Build the project

### Setup Instructions

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
uv sync --all-extra
```
