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
import sys
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

from dotenv import load_dotenv
from litellm import acompletion
from litellm.exceptions import BadRequestError
from openai import RateLimitError

load_dotenv()

EXPONENTIAL_BASE = 2


T = TypeVar("T", str, List[str])


class BaseModel(ABC, Generic[T]):
    @abstractmethod
    async def predict(self, prompt: str) -> T:
        pass


class LiteLLM_Model(BaseModel[str]):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        temp: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        max_retries: int = 3,
    ):
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
                try:
                    response = await acompletion(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temp,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        logger_fn=None,
                    )
                except BadRequestError:
                    logging.error(
                        f"Invalid model name: {self.model_name} or the api key for the model is invalid"
                    )
                    sys.exit(1)
                    raise Exception(
                        f"Invalid model name: {self.model_name} or the api key for the model is invalid"
                    )
                    sys.exit(1)
                except Exception as e:
                    logging.error(f"Error in request: {e}")
                    raise Exception(f"Error in request: {e}")
                if response.choices[0].message.content is not None:
                    return response.choices[0].message.content
                else:
                    logging.debug("No content in response" + str(response))
                    raise Exception("No content in response")

            except RateLimitError as e:
                delay *= EXPONENTIAL_BASE * (1 + random.random())
                logging.warning(
                    f"RateLimitError, retrying after {round(delay, 2)} seconds, {i + 1}-th retry...",
                    e,
                )
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                logging.warning(f"Error in retry {i + 1}, retrying...", e)
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
