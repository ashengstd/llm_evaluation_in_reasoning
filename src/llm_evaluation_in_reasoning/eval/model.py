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

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

from dotenv import load_dotenv
from litellm import acompletion
from openai import RateLimitError

load_dotenv()

# Map model names to their corresponding model IDs, see it in litellm docs: https://docs.litellm.ai/docs/providers
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

EXPONENTIAL_BASE = 2


T = TypeVar("T", str, List[str])


class BaseModel(ABC, Generic[T]):
    @abstractmethod
    async def predict(self, prompt: str) -> T:
        pass


class EvalModel(BaseModel[str]):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        temp: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        max_retries: int = 3,
    ):
        if model_name not in MODEL_MAP:
            raise ValueError(f"Invalid model name: {model_name}")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temp = None if "o1" in model_name else temp
        self.max_tokens = None if "o1" in model_name else max_tokens
        self.top_p = None if "o1" in model_name else top_p
        self.max_retries = max_retries

    async def predict(self, prompt: str) -> str:
        delay = 2.0

        for i in range(self.max_retries):
            try:
                messages = []
                if self.system_prompt is not None:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})
                logging.debug(f"Sending prompt to model: {prompt}")
                response = await acompletion(
                    model=MODEL_MAP[self.model_name],
                    messages=messages,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    logger_fn=None,
                )

                if response.choices[0].message.content is not None:
                    return response.choices[0].message.content
                else:
                    logging.debug("No content in response" + str(response))
                    raise Exception("No content in response")

            except RateLimitError as e:
                delay *= EXPONENTIAL_BASE * (1 + random.random())
                logging.warning(
                    f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...",
                    e,
                )
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                logging.warning(f"Error in retry {i+1}, retrying...", e)
                continue
        logging.error(f"Failed to get response after {self.max_retries} retries")
        raise Exception("Failed to get response after max retries")


class MajorityVoteModel(BaseModel[List[str]]):
    def __init__(self, model: BaseModel[str], num_responses: int = 3):
        self.model = model
        self.num_responses = num_responses

    async def predict(self, prompt: str) -> List[str]:
        tasks = [self.model.predict(prompt) for _ in range(self.num_responses)]
        return await asyncio.gather(*tasks)
