"""
This file contains the Generator classes and generator factory.
"""

import os
import time
from openai import OpenAI

OpenAIModelList = ["gpt-4o", "gpt-4o-mini", "gpt-4o-v", "gpt-4o-mini-v", "o3-2025-04-16", "gpt-4o-mini-2024-07-18"]

def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]


class Generator:

    def __init__(self, model: str, verbose: bool = False):
        self.model = model
        self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        self.verbose = verbose

    def __call__(self, messages):
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        max_retries = 5
        retry_count = 0
        fatal = False
        fatal_reason = None
        while retry_count < max_retries:
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                self.total_tokens += result.usage.total_tokens
                self.input_tokens += result.usage.prompt_tokens
                self.output_tokens += result.usage.completion_tokens
                break # break out of while loop if no error
            
            except Exception as e:
                print(f"An error occurred: {e}.")
                if "context_length_exceeded" in f"{e}" or "too long" in f"{e}":
                    fatal = True
                    fatal_reason = f"{e}"
                    break
                else:
                    print("Retrying...")
                    retry_count += 1
                    time.sleep(10 * retry_count)  # Wait 
        
        if fatal:
            print(f"Fatal error occured. ERROR: {fatal_reason}")
            res = f"Fatal error occured. ERROR: {fatal_reason}"
        elif retry_count == max_retries:
            print("Max retries reached. Skipping...")
            res = "Max retries reached. Skipping..."
        else:
            try:
                res = result.choices[0].message.content
            except Exception as e:
                print("Error:", e)
                res = ""
            #print(res)
        return res
