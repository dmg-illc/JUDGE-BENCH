# LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks

Paper link: https://aclanthology.org/2025.acl-short.20/

![Evaluation](overview_meta_eval.png)

## Data

Check the [data](https://github.com/dmg-illc/JUDGE-BENCH/tree/master/data) directory for preparing the test data.

## Model responses

The script `prompt_model_json.py` automatically runs an LLM on a(ll) dataset(s) and stores the model's responses. It uses the prompt that was provided with the dataset, and allows for additions to this prompt e.g. to request short answers from the model. It stores a .json file per model and dataset, which contains model responses and details of the run (e.g. batch size, additional prompts used) in addition to all information from the input data file. Example usage:

```
python prompt_model_json.py -d cola -t YOUR_HF_TOKEN -m meta-llama/Meta-Llama-3.1-70B-Instruct -b 8 -nt 25 -ap 1 -rd results
```
If you do not specify a dataset, model responses are generated for all datasets in Judge-Bench:

```
python prompt_model_json.py -t YOUR_HF_TOKEN -m meta-llama/Meta-Llama-3.1-70B-Instruct -b 8 -nt 25 -ap 1 -rd results
```

### Datasets
Currently, the dataset arguments corresponding to all datasets in Judge-Bench are:
```
{"cola","cola-grammar","dailydialog-acceptability","inferential-strategies","llmbar-adversarial","llmbar-natural",
"medical-safety","newsroom","persona_chat","qags","recipe_crowd_sourcing_data","roscoe-cosmos","roscoe-drop",
"roscoe-esnli","roscoe-gsm8k","summeval","switchboard-acceptability","topical_chat","toxic_chat-train",
"toxic_chat-test","wmt-human_en_de","wmt-human_zh_en","wmt-23_en_de","wmt-23_zh_en","dices_990","dices_350_expert",
"dices_350_crowdsourced"}
```
### Models
The arguments corresponding to models that have been evaluated on Judge-Bench are:
```
{"berkeley-nest/Starling-LM-7B-alpha","mistralai/Mistral-7B-Instruct-v0.3","meta-llama/Meta-Llama-3.1-8B-Instruct",
"meta-llama/Meta-Llama-3.1-70B-Instruct","CohereForAI/c4ai-command-r-v01","CohereForAI/c4ai-command-r-plus",
"mistralai/Mixtral-8x22B-Instruct-v0.1","mistralai/Mixtral-8x7B-Instruct-v0.1","allenai/OLMo-7B-0724-Instruct-hf",
"gpt-4o", "gemini-1.5-flash-latest"}
```
The open-source models are obtained through HuggingFace, so obtaining results for a different model that is available on HuggingFace can be done simply by specifying its name as the model argument. In order to obtain responses from a different proprietary model, the `APIModel` class in `models.py` might need small modifications depending on the specific API.

## Evaluation

Next, the script `eval_responses.py` computes the correlation metrics from the human evaluations and model responses stored at the previous step. Depending on the type of judgments (graded, categorical), the relevant metrics are computed (correlation scores for graded judgments and Cohen's Kappa for categorical ones). This script also computes the agreement between humans (Krippendorff's Alpha) for datasets with multiple judgments per instance. Example usage:

```
python eval_responses.py -rd results -d cola -m meta-llama/Meta-Llama-3.1-70B-Instruct
```
This script saves a .json file for each dataset, which can be used to reproduce the tables and figures from the paper in `results_notebook.ipynb`.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{bavaresco-etal-2025-llms,
    title = "{LLM}s instead of Human Judges? A Large Scale Empirical Study across 20 {NLP} Evaluation Tasks",
    author = "Bavaresco, Anna  and
      Bernardi, Raffaella  and
      Bertolazzi, Leonardo  and
      Elliott, Desmond  and
      Fern{\'a}ndez, Raquel  and
      Gatt, Albert  and
      Ghaleb, Esam  and
      Giulianelli, Mario  and
      Hanna, Michael  and
      Koller, Alexander  and
      Martins, Andre  and
      Mondorf, Philipp  and
      Neplenbroek, Vera  and
      Pezzelle, Sandro  and
      Plank, Barbara  and
      Schlangen, David  and
      Suglia, Alessandro  and
      Surikuchi, Aditya K  and
      Takmaz, Ece  and
      Testoni, Alberto",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-short.20/",
    pages = "238--255",
    ISBN = "979-8-89176-252-7",
    abstract = "There is an increasing trend towards evaluating NLP models with LLMs instead of human judgments, raising questions about the validity of these evaluations, as well as their reproducibility in the case of proprietary models. We provide JUDGE-BENCH, an extensible collection of 20 NLP datasets with human annotations covering a broad range of evaluated properties and types of data, and comprehensively evaluate 11 current LLMs, covering both open-weight and proprietary models, for their ability to replicate the annotations. Our evaluations show substantial variance across models and datasets. Models are reliable evaluators on some tasks, but overall display substantial variability depending on the property being evaluated, the expertise level of the human judges, and whether the language is human or model-generated. We conclude that LLMs should be carefully validated against human judgments before being used as evaluators."
}
