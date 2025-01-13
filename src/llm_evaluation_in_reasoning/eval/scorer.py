# Modified by ashengstd on 2025.01.10
# MIT License

# Copyright (c) 2024 simple-bench

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import re
from typing import List

from llm_evaluation_in_reasoning.data.question import QuestionType


def extract_answer(output: str, question_type: QuestionType) -> str | int:
    match question_type:
        case QuestionType.MULTIPLE_CHOICE:
            output = output.strip()
            match = re.search(r"Final Answer:\s*([A-F])", output, re.IGNORECASE)
        case QuestionType.BLANK_FILL:
            output = output.replace(",", " ")
            match = re.findall(r"-?\d+\.?\d*", output)[-1]
        case _:
            logging.error("Invalid question type")
            raise ValueError("Invalid question type")

    if match:
        answer = (
            match.group(1).upper()
            if question_type == QuestionType.MULTIPLE_CHOICE
            else int(float(str(match)))
        )
        logging.info(f"Answer extracted: {answer}")
        return answer

    logging.error("No answer found in model output")
    raise ValueError("No answer found in model output")


def eval_majority_vote(
    output: List[str], answer: str | int, question_type: QuestionType
) -> bool:
    model_answers = []
    for _output in output:
        try:
            model_answers.append(extract_answer(_output, question_type=question_type))
        except ValueError:
            logging.warning("no answer in this response")
            continue  # Skip this output if extraction fails

    if not model_answers:
        logging.error("Failed to extract any valid answers from model outputs")
        raise ValueError("Failed to extract any valid answers from model outputs")

    return model_answers.count(answer) > len(model_answers) / 2


def eval_question_once(
    output: str, answer: str | int, question_type: QuestionType
) -> bool:
    model_answer = extract_answer(output, question_type)
    logging.info(f"Model answer: {model_answer} | Ground truth answer: {answer}")
    return model_answer == answer
