# NEWSROOM Dataset

Human annotated dataset from "Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies" ([Grusky et al., NAACL 2018](https://doi.org/10.18653/v1/N18-1065)).

Annotations judge the summary quality across two semantic dimensions: informativeness and relevancy, and two syntactic dimensions: fluency and coherence. Evaluation was performed on 60 summaries generated by various summarization systems for articles from the [Newsroom dataset](https://github.com/lil-lab/newsroom/tree/master/humaneval). 

Human judgements are crowd-sourced on Amazon Mechanical Turk. Each summary is judged by 3 annotators.

## Download Original Data
To convert the data, please first download the original data by running:
```
bash download_original_data.sh
```

## Convert Data
Then, convert the data by running:
```
python convert.py
```

The converted data will be saved as json file in the current directory.