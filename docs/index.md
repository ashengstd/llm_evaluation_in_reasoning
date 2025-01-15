---
title: LLM_EVAL
layout: home
nav_order: 1
---

# llm_evaluation_in_reasoning

[![example workflow](https://github.com/ashengstd/llm_evaluation_in_reasoning/actions/workflows/publish-pypi-release.yml/badge.svg)](https://github.com/ashengstd/llm_evaluation_in_reasoning) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm_evaluation_in_reasoning)](https://pypi.org/project/llm_evaluation_in_reasoning) [![PyPI](https://img.shields.io/pypi/v/llm_evaluation_in_reasoning.svg)](https://pypi.org/project/llm_evaluation_in_reasoning/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/llm_evaluation_in_reasoning.svg)](https://pypi.org/project/llm_evaluation_in_reasoning/) [![GitHub License](https://img.shields.io/github/license/ashengstd/llm_evaluation_in_reasoning)](https://github.com/ashengstd/llm_evaluation_in_reasoning) [![GitHub Release](https://img.shields.io/github/v/release/ashengstd/llm_evaluation_in_reasoning)](https://github.com/ashengstd/llm_evaluation_in_reasoning)

## Overview

`llm_evaluation_in_reasoning` is a project designed to evaluate the reasoning capabilities of large language models (LLMs). It supports multiple reasoning benchmarks, including GSM-Symbolic, GSM8K, MMLU, and SimpleBench. This project helps you assess the performance of various models and understand their reasoning skills.

## Installation

### Step 1: Install the Package

To get started, you need to install the package from PyPI:

```shell
pip install llm_evaluation_in_reasoning
```

### Step 2: Create the `.env` File

Create a `.env` file in the root directory of your project with the following content:

```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
...
```

The API keys you provide will be used to fetch the valid models supported by `Litellm`. Make sure to replace `<your key>` with actual API keys from the respective platforms.

## Run the Evaluation

This project supports several evaluation benchmarks:

- **GSM-Symbolic**
- **GSM8K**
- **MMLU**
- **SimpleBench**

To run a benchmark, use the following command:

```shell
llm_eval --model_name=ollama/qwen2.5:0.5b --dataset=SimpleBench
```

Run `llm_eval --help` for more details and options.

## Supported Models

The model support is based on `Litellm`, which provides integrations with different LLM providers. You can check the full list of supported providers in the [Litellm Providers Documentation](https://docs.litellm.ai/docs/providers).

## Building the Project

### Step 1: Clone the Repository

To start developing or contributing to the project, clone the GitHub repository and navigate into the project folder:

```shell
git clone https://github.com/ashengstd/llm_evaluation_in_reasoning.git
cd llm_evaluation_in_reasoning
```

### Step 2: Install Dependencies with `uv`

The recommended way to install project dependencies is by using `uv`. If you don't have it installed, follow these steps:

#### macOS and Linux:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Sync the Dependencies

After installing `uv`, sync the dependencies with:

```shell
uv sync --all-extra
```

This will ensure that all required dependencies are installed and up-to-date.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/ashengstd/llm_evaluation_in_reasoning/blob/main/LICENSE) file for more details.

---

For more information, check out the official documentation or contribute to the repository. We welcome pull requests and issue reports!
