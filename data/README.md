# Data format

For now, we're keeping each crowdsourced dataset in a separate JSON file that follows the JSON schema in [schema.json]([https://github.com/coli-saar/llm-meta-evaluation/blob/main/data/schema.json](https://github.com/dmg-illc/TextJudge/blob/master/data/example.json)). You can check your own JSON file against the schema using e.g. [check-jsonschema](https://github.com/python-jsonschema/check-jsonschema).

## Overall file structure

The dataset is structured as follows:

- `dataset`, `dataset_url`: General information about where the dataset came from.
- `expert_annotator`: Whether the annotation was crowdsourced or not (every non-crowdsourced annotation was considered ad expert)
- `annotations`: Declares the categories with which each instance in the dataset was annotated. In [example.json](https://github.com/coli-saar/llm-meta-evaluation/blob/main/data/example.json), we collected annotations for `grammaticality` and `fluency` on a 1-7 scale. For each annotation, we specify the worst and best possible annotation value. There is also a prompt, see below.
- `instances`: Lists the instances for which we collected human judgments. In each instance, the `instance` field contains the sentence/text that was annotated, and `annotations` contains the annotations for each annotation category - both the mean of the human judgments and the list of scores that the individual humans gave.


## Types of variables and aggregation method

The annotations were divided into 3 main types, which are specified within the `category` field:
- `continuous`: Annotations that could be expressed with any value within a given range. These were aggregated with an arithmetic mean 
- `graded`: Annotations that could be expressed with a fixed set of values within a given range (e.g., 0 for "ungrammatical", 1 for "poorly formed", ..., 5 for "fully grammatical"). These were aggregated with an arithmetic mean
- `categorical`: Annotations with binary labels (e.g., True or False) or labels referring to non-ordered taxonomies (e.g. 1 for "swearing", 2 for "insult", 3 for "sexism", etc.). These annotations were aggregated by majority vote

## Prompts

The `prompt` field in the `annotations` declarations is a free text that describes what should be annotated. This can initially be the literal text that we showed to the human crowdworkers, but perhaps it can over time be replaced by better prompts that are specific to an LLM. 

The evaluation script should interpret the `prompt` field as a string with placeholders. For now, I (Alexander) have assumed that it will be evaluated as a [jinja](https://palletsprojects.com/p/jinja/) template, with the `instance` variable assigned the value of the `instance` field of each test instance. Thus, all occurrences of the string `{{ instance }}` will be replaced by the actual sentences/texts that were annotated. Of course, we could change this.

## Licenses
Please find an overview of the licenses for each dataset in the table below.

| Dataset            | License                  |
|--------------------|--------------------------|
|[CoLA](https://nyu-mll.github.io/CoLA/)|[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt)|
|[CoLA grammar](https://nyu-mll.github.io/CoLA/#grammatical_annotations)| [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt)|
|[Switchboard](https://data.cstr.ed.ac.uk/sarenne/INTERSPEECH2022/)|N/A|
|[Dailydialog](https://data.cstr.ed.ac.uk/sarenne/INTERSPEECH2022/)|N/A|
|[Inferential strategies](https://huggingface.co/datasets/mainlp/inferential_strategies)|[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt)|
|[ROSCOE](https://github.com/facebookresearch/ParlAI/tree/main/projects/roscoe)|[MIT](https://github.com/facebookresearch/ParlAI/tree/main?tab=MIT-1-ov-file)|
|[Recipe-generation](https://github.com/interactive-cookbook/recipe-generation)|[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt)|
|[Medical-safety](https://github.com/GavinAbercrombie/medical-safety)|[GPL-3.0 license](https://github.com/GavinAbercrombie/medical-safety?tab=GPL-3.0-1-ov-file)|
|[DICES](https://github.com/google-research-datasets/dices-dataset)|[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt)|
|[ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat)|[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt)|
|[Topical Chat](http://shikib.com/usr)|N/A|
|[Persona Chat](http://shikib.com/usr)|N/A|
|[WMT 20 EnDe](https://github.com/google/wmt-mqm-human-evaluation/tree/main/newstest2020)|[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)|
|[WMT 20 ZhEn](https://github.com/google/wmt-mqm-human-evaluation/tree/main/newstest2020)|[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)|
|[WMT 23 EnDe](https://github.com/google-research/mt-metrics-eval)|[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)|
|[WMT 23 ZhEn](https://github.com/google-research/mt-metrics-eval)|[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)|
|[G-Eval / SummEval](https://github.com/nlpyang/geval)|[MIT](https://github.com/nlpyang/geval?tab=MIT-1-ov-file)|
|[QAGS](https://github.com/W4ngatang/qags/tree/master)|N/A|
|[NewsRoom](https://github.com/lil-lab/newsroom/tree/master/humaneval)|[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)|
|[LLMBar](https://github.com/princeton-nlp/LLMBar)|[MIT](https://github.com/princeton-nlp/LLMBar?tab=MIT-1-ov-file)|
