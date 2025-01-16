---
title: BaseBenchDataloader Documentation
layout: default
parent: API
nav_layout: 1
---

# BaseBenchDataloader

The `BaseBenchDataloader` class serves as an abstract base class for implementing benchmarking dataloaders used in evaluating the reasoning capabilities of models. This documentation explains the class structure, its methods, and their usage.

## Class Overview

### Attributes

- **`dataset`**: A `Dataset` or `DatasetDict` object containing the questions and answers used for evaluation.
- **`progress_bar`**: A `rich.progress.Progress` instance used for tracking evaluation progress.
- **`question_key`**: The key in the dataset used for accessing questions.
- **`answer_key`**: The key in the dataset used for accessing answers.

### Constructor

```python
BaseBenchDataloader()
```

Initializes a new instance of the `BaseBenchDataloader` class. This is an abstract class and cannot be instantiated directly.

## Methods

### Static Methods

#### `initial_default_prompt()`

Returns the default system prompt for the evaluation.

```python
@staticmethod
def initial_default_prompt() -> str:
    return ""
```

#### `eval_single_question(predicted_answer: str | int, answer: str | int) -> bool`

Evaluates the correctness of a single predicted answer.

- **Parameters:**
  - `predicted_answer`: The answer predicted by the model.
  - `answer`: The correct answer from the dataset.
- **Returns:**
  - `bool`: Whether the prediction is correct.

#### `vote_majority(output: List[str | int], answer: str | int) -> bool`

Implements majority voting for multiple outputs from the model.

- **Parameters:**
  - `output`: List of model outputs.
  - `answer`: The correct answer.
- **Returns:**
  - `bool`: Whether the majority voted correctly.

### Abstract Methods

#### `process_question(example: dict) -> dict`

Processes a question from the dataset. This method must be implemented in derived classes.

#### `extract_answer(output: str) -> str | int`

Extracts the answer from the model's output. This method must be implemented in derived classes.

### Asynchronous Methods

#### `evaluate_model(model: LiteLLM_Model | MajorityVoteModel, system_prompt: str | None = None) -> tuple[List[dict], float]`

Evaluates the performance of a model on the dataset.

- **Parameters:**
  - `model`: The model instance to evaluate. It can be either `LiteLLM_Model` or `MajorityVoteModel`.
  - `system_prompt`: An optional system prompt string. If not provided, the default prompt is used.
- **Returns:**
  - `tuple[List[dict], float]`: A tuple containing:
    - `results`: A list of dictionaries for each evaluation entry with keys for the question, response, correct answer, and correctness.
    - `accuracy`: The overall accuracy of the model.

### Usage Example

#### Example Workflow

```python
from my_project.bench_dataloader import BaseBenchDataloader
from llm_evaluation_in_reasoning.eval.model import LiteLLM_Model

class MyDataloader(BaseBenchDataloader):
    def process_question(self, example):
        # Custom processing logic
        pass

    def extract_answer(self, output):
        # Custom answer extraction
        pass

    def eval_single_question(self, predicted_answer, answer):
        return predicted_answer == answer

    def vote_majority(self, output, answer):
        return output.count(answer) > len(output) / 2

# Instantiate and evaluate
my_loader = MyDataloader()
model = LiteLLM_Model()
results, accuracy = await my_loader.evaluate_model(model)
print(f"Accuracy: {accuracy:.2%}")
```

## Logging and Error Handling

### Logging

This class uses the Python `logging` module to log progress, debug information, and errors during evaluation.

### Error Handling

Errors encountered during the evaluation process are logged and stored in the `results` list with the associated question and error details.

## Dependencies

- `logging`: Standard Python logging library.
- `rich`: Library for rich text and progress bar visualization.
- `datasets`: Library for loading and processing datasets.
- `LiteLLM_Model`, `MajorityVoteModel`: Model classes used for evaluation.

For more details, refer to the [GitHub repository](https://github.com/your-repo-link).
