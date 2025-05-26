from transformers import (
    AutoTokenizer,
    pipeline,
)
import torch
from tqdm import tqdm
import time

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class HFModel:
    def __init__(self, name, new_tokens) -> None:
        self.model = pipeline(
            "text-generation",
            model=name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True if "OLMo" in name else False,
        )
        if not self.model.tokenizer.pad_token_id:
            self.model.tokenizer.pad_token_id = (
                self.model.tokenizer.eos_token_id
            )
        self.n_tokens = new_tokens

    def process(self, example):
        user_message = {"role": "user", "content": example}
        messages = [user_message]
        return self.model.tokenizer.apply_chat_template(
            messages, tokenize=False
        )

    def generate_responses(self, dataset, batch_size):
        dataset = [self.process(example) for example in dataset]
        results = []
        for n_batch in tqdm(range(len(dataset) // batch_size + 1)):
            batch = dataset[batch_size * n_batch : batch_size * (n_batch + 1)]
            if len(batch) == 0:
                continue
            responses = self.model(
                batch,
                batch_size=batch_size,
                max_new_tokens=self.n_tokens,
                do_sample=False,
                num_beams=1,
                return_full_text=False,
            )
            for response in responses:
                results.append(response[0]["generated_text"])
        return results


class APIModel:
    name: str
    new_tokens: int
    family: str

    def __init__(self, name: str, new_tokens: int) -> None:
        self.name = name
        self.new_tokens = new_tokens
        if "gpt" in name:
            self.family = "OpenAI"
            self.client = OpenAI()
        elif "claude" in name:
            self.family = "Anthropic"
            self.client = Anthropic()
        elif "gemini" in name:
            self.family = "Google"
            self.client = genai.GenerativeModel(name)
        else:
            raise ValueError(
                f'Model "{name} is not a valid ClosedModel (gpt, claude, gemini)'
            )

    def generate_responses(self, dataset, batch_size):
        responses = []
        for query in tqdm(dataset):
            user_message = {"role": "user", "content": query}

            messages = [user_message]

            if self.family == "OpenAI":
                response = (
                    self.client.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        max_tokens=self.new_tokens,
                        temperature=0,
                    )
                    .choices[0]
                    .message.content
                )
                responses.append(response)

            elif self.family == "Anthropic":
                response = (
                    self.client.messages.create(
                        model=self.name,
                        messages=messages,
                        max_tokens=self.new_tokens,
                        temperature=0,
                    )
                    .content[0]
                    .text
                )
                responses.append(response)
                # time.sleep(3) # in case of rate limiting resort to 15 RPM

            elif self.family == "Google":
                response = self.client.generate_content(
                    query,
                    generation_config={
                        "max_output_tokens": self.new_tokens,
                        "temperature": 0,
                    },
                    safety_settings={  # https://ai.google.dev/gemini-api/docs/safety-settings (e.g., usecase: QAGS dataset)
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                try:
                    responses.append(response.text)
                except ValueError:
                    print(f"FAILED REQUEST: {messages}\nRESPONSE: {response}")
                    responses.append("")  # to keep all valid responses

            else:
                raise ValueError(
                    f'Family "{self.family} is not a valid family'
                )

        return responses
