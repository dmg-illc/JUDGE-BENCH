# Inferential Strategies Dataset

Human annotated dataset from "Comparing Inferential Strategies of Humans and Large Language Models in Deductive Reasoning" ([Mondorf and Plank, ACL 2024](https://doi.org/10.48550/arXiv.2402.14856)).

Responses of the following language models are evaluated:
1. [Llama-2-chat-hf3 (7B, 13B, and 70B)](https://huggingface.co/meta-llama)
2. [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
3. [Zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

Annotations judge the logical validity of the models' reasoning when solving problems of propositional logic. Binary labels are assigned to each response, indicating whether the rationale provided by the model is sound (True) or not (False). Each model is assessed on 12 problems of propositional logic across 5 random seeds, resulting in a total of 60 responses per model.

Each rationale is judged by 2 expert annotators.

## Convert Data
Then, convert the data by running:
```
python convert.py
```

The converted data will be saved as json file in the current directory.