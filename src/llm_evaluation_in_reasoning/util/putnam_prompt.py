from typing import Dict, List

FEW_SHOT_EXAMPLES = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]

BASE_FEW_SHOT_COT = "\n\n".join(
    [
        f"Problem:\n{ex['problem']}\n\nSolution:\nLet's think step by step {ex['solution']}"
        for ex in FEW_SHOT_EXAMPLES
    ]
)
BASE_FEW_SHOT = "\n\n".join(
    [
        f"Problem:\n{ex['problem']}\n\nSolution:\n {ex['solution']}"
        for ex in FEW_SHOT_EXAMPLES
    ]
)

# base model prompts
general_zero_shot_prompt = "Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}."
general_few_shot_prompt = f"Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}. {BASE_FEW_SHOT}"
general_putnam_few_shot_prompt = f"Given a problem from the William Lowell Putnam Mathematical Competition, compose a detailed solution. Always give the final answer inside a \\boxed{{answer}}. {BASE_FEW_SHOT}"
general_few_shot_cot_prompt = f"Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}. {BASE_FEW_SHOT_COT}"
general_putnam_few_shot_cot_prompt = f"Given a problem from the William Lowell Putnam Mathematical Competition, compose a detailed solution. Always give the final answer inside a \\boxed{{answer}}. {BASE_FEW_SHOT_COT}"


# instruct model prompts
mistral_instruct_zero_shot_prompt = "<s> [INST] Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}."
mistral_instruct_few_shot_prompt = f"<s> [INST] You are a skilled and intelligent mathematician AI for tackling challenging mathematics questions. Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}. {BASE_FEW_SHOT}"

LLAMA_FEW_SHOT = "\n\n".join(
    [
        f"<|start_header_id|>user<|end_header_id|> Problem:\n{ex['problem']}"
        f"\n\n <|start_header_id|>assistant<|end_header_id| Solution:\nLet's think step by step {ex['solution']}"
        for ex in FEW_SHOT_EXAMPLES
    ]
)
llama_instruct_zero_shot_prompt = "<|start_header_id|>system<|end_header_id|> You are a skilled and intelligent mathematician AI for tackling challenging mathematics questions <|eot_id|>  <|start_header_id|>user<|end_header_id|> Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}"
llama_instruct_few_shot_prompt = f"<|start_header_id|>system<|end_header_id|> You are a skilled and intelligent mathematician AI for tackling challenging mathematics questions <|eot_id|> <|start_header_id|>user<|end_header_id|> Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}. Here are some examples of questions{LLAMA_FEW_SHOT}"

GEMMA_FEW_SHOT = "\n\n".join(
    [
        "<start_of_turn>user"
        f"Problem:\n{ex['problem']}<end_of_turn>"
        "<start_of_turn>model"
        f"\n\nSolution:\nLet's think step by step {ex['solution']}<end_of_turn>"
        for ex in FEW_SHOT_EXAMPLES
    ]
)
gemma_instruct_zero_shot_prompt = "<start_of_turn>user Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}"
gemma_instruct_few_shot_prompt = f"<start_of_turn>user Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}} Here are some examples of questions{GEMMA_FEW_SHOT} <start_of_turn>user"


def generate_prompt(prompt_name: str, problems: List[Dict[str, str]]) -> List[str]:
    """
    Generate prompts for the given prompt name and problems.
    The prompt contains the problem it self.
    params:
        prompt_name: str: Name of the prompt
        problems: List[Dict[str, str]]: List of problems
    """
    type = prompt_name.split("_")[0]
    prompt_string = globals()[prompt_name]
    prompt = []

    if type == "general":
        prompt = [
            prompt_string
            + f"\n\nProblem:\n{p['problem']}\n\nSolution:\nLet's think step by step."
            for p in problems
        ]
    elif type == "mistral":
        prompt = [
            prompt_string
            + f"\n\nProblem:\n{p['problem']}\n\nSolution:\nLet's think step by step. [/INST]"
            for p in problems
        ]
    elif type == "llama":
        prompt = [
            prompt_string
            + f"\n\nProblem:\n{p['problem']}\n\nSolution:\nLet's think step by step. <|start_header_id|>assistant<|end_header_id|>"
            for p in problems
        ]
    elif type == "gemma":
        prompt = [
            prompt_string
            + f"\n\nProblem:\n{p['problem']}\n\nSolution:\nLet's think step by step.<end_of_turn> <start_of_turn>model"
            for p in problems
        ]
    print(len(prompt))
    return prompt


def convert_prompt(example):
    example["problem"] = example["problem"].replace("\n", " ")
