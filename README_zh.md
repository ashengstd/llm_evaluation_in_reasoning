# llm_evaluation_in_reasoning

[![example workflow](https://github.com/ashengstd/llm_evaluation_in_reasoning/actions/workflows/publish-pypi-release.yml/badge.svg)](https://github.com/ashengstd/llm_evaluation_in_reasoning) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm_evaluation_in_reasoning)](https://pypi.org/project/llm_evaluation_in_reasoning) [![PyPI](https://img.shields.io/pypi/v/llm_evaluation_in_reasoning.svg)](https://pypi.org/project/llm_evaluation_in_reasoning/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/llm_evaluation_in_reasoning.svg)](https://pypi.org/project/llm_evaluation_in_reasoning/) [![GitHub License](https://img.shields.io/github/license/ashengstd/llm_evaluation_in_reasoning)](https://github.com/ashengstd/llm_evaluation_in_reasoning) [![GitHub Release](https://img.shields.io/github/v/release/ashengstd/llm_evaluation_in_reasoning)](https://github.com/ashengstd/llm_evaluation_in_reasoning)

一个用于评估 LLM 推理能力的项目。
**其他语言版本: [English](https://github.com/ashengstd/llm_evaluation_in_reasoning/blob/main/README.md), [中文](https://github.com/ashengstd/llm_evaluation_in_reasoning/blob/main/README_zh.md).**

## 运行评估

### 安装软件包

```shell
pip install llm_evaluation_in_reasoning
```

### 创建 .env 文件

创建一个 .env 文件，内容如下：

```
OPENAI_API_KEY=<你的密钥>
ANTHROPIC_API_KEY=<你的密钥>
...
```

提供的 API 密钥将用于获取由 Litellm 支持的有效模型。

### 运行说明

支持的数据集包括：GSM-Symbolic、GSM8K、MMLU、SimpleBench。
运行基准测试：

```shell
llm_eval --model_name=ollama/qwen2.5:0.5b --dataset=SimpleBench # 运行 llm_eval --help 查看帮助信息
```

## 模型支持

模型支持基于 Litellm，请参阅文档：[Litellm Providers](https://docs.litellm.ai/docs/providers)

## 构建项目

### 设置

克隆 GitHub 仓库并进入目录。

```shell
git clone https://github.com/ashengstd/llm_evaluation_in_reasoning.git
cd llm_evaluation_in_reasoning
```

### 安装 uv

安装依赖的最佳方式是使用 uv。如果你的环境中尚未安装，可以通过以下方式安装：

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh # macOS 和 Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

### 同步依赖

```shell
uv sync --all-extra
```
