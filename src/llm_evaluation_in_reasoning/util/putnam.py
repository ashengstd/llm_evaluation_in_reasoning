import logging
import re
from typing import Optional

from sympy import pretty, simplify, sympify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def ground_truth_boxed_answer(solution: str) -> str:
    answer = last_boxed_only_string(solution)
    if answer:
        return normalize_final_answer(remove_boxed(answer))
    else:
        logging.error("No boxed answer found for problem!")
        raise ValueError("No boxed answer found for problem!")


def convert_answer(example):
    example["answer"] = ground_truth_boxed_answer(example["solution"])
    return example


def normalize_latex_expression(latex_str: str) -> str:
    # 替换 LaTeX 符号 \times 为 *
    latex_str = latex_str.replace(r"\times", "*")

    # 替换 LaTeX 符号 \^{n} 为 **n
    latex_str = re.sub(r"\\\^(\{[0-9]+\})", r"**\1", latex_str)

    return latex_str


def simplify_answer(answer):
    answer = parse_answer(answer)
    try:
        answer = simplify(answer)
    except Exception as e:
        logging.error(f"Error simplifying answer: {e}")
    return answer


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        # r"Final Answer: The final answer is(.*?).",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        # print('++++++++++++++++++++++++++')
        # print(text)
        # print('++++++++++++++++++++++++++')
        return INVALID_ANSWER


def get_generated_answer(result: str) -> str:
    unnormalized_answer = get_unnormalized_answer(result)
    answer = normalize_final_answer(unnormalized_answer)

    return answer


def parse_answer(answer):
    ret = answer
    answer = answer.lower()
    answer = simplify_algebra(answer)
    answer = simplify_frac(answer)
    answer = simplify_sqrt(answer)
    answer = simplify_power(answer)
    answer = answer.replace("\\log", "log")
    answer = answer.replace("\\pi", "pi")
    answer = answer.replace("\\ln", "ln")

    try:
        transformations = standard_transformations + (
            implicit_multiplication,
            convert_xor,
        )
        ret = parse_expr(answer, transformations=transformations)
    except Exception as e:
        logging.error(f"Error parsing answer: {e}")
    return ret


def simplify_frac(answer):
    pattern = re.compile(r"\\frac\{(.*?)\}\{(.*?)\}")

    # Function to replace each match
    def replacer(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return f"({numerator}) / ({denominator})"

    # Perform the replacement
    return re.sub(pattern, replacer, answer)


def simplify_power(answer):
    # This pattern matches the base and exponent parts of the power expression
    pattern = re.compile(r"(\w+|\(.+?\))\^{([^{}]+)}")

    def replacer(match):
        base = match.group(1)
        exponent = match.group(2)

        # Wrap base in parentheses if it's not a single word character
        if not re.fullmatch(r"\w+", base):
            base = f"({base})"

        # Wrap exponent in parentheses if it's not a single word character
        if not re.fullmatch(r"\w+", exponent):
            exponent = f"({exponent})"

        return f"{base}^{exponent}"

    return re.sub(pattern, replacer, answer)


def simplify_sqrt(expression):
    # Define the regular expression pattern for \sqrt{X}
    pattern = re.compile(r"\\sqrt\{(.*?)\}")

    # Function to replace each match
    def replacer(match):
        content = match.group(1)
        return f"sqrt({content})"

    # Perform the replacement
    replaced_expression = re.sub(pattern, replacer, expression)

    return replaced_expression


# def simplify_algebra(answer):
#     try:
#         answer = pretty(
#             simplify(sympify(parse_expr(answer, transformations="all"))),
#             use_unicode=False,
#         )
#     except Exception as e:
#         logging.error(f"Error simplifying algebra: {e}")
#     return answer


def simplify_algebra(answer):
    logging.info(f"Original answer: {answer}")

    try:
        # Step 1: Parse the answer to a sympy expression
        parsed_expr = parse_expr(answer, transformations="all")
        logging.info(f"Parsed expression: {parsed_expr}")

        # Step 2: Simplify the parsed expression
        simplified_expr = simplify(sympify(parsed_expr))
        logging.info(f"Simplified expression: {simplified_expr}")

        # Step 3: Get the pretty formatted version
        answer = pretty(simplified_expr, use_unicode=False)
        logging.info(f"Pretty formatted answer: {answer}")

    except Exception as e:
        logging.error(f"Error simplifying algebra: {e}")
        logging.exception("Exception details:")
        return None  # Return None to indicate an error occurred
    return answer
