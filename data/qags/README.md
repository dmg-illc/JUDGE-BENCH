# Question Answering and Generation for Summarization (QAGS) Dataset

Human annotated dataset from "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries" ([Wang et al., ACL 2020](https://doi.org/10.18653/v1/2020.acl-main.450)).

Annotations judge the factual consistency of one-sentence summaries with respect to the reference article. The summaries and articles are collected from two datasets: [CNN/DailyMail](https://proceedings.neurips.cc/paper_files/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html) and [XSum](https://doi.org/10.18653/v1/D18-1206). Annotation protocols differ slightly for the two datasets, which is accounted for in the conversion.

Human judgements are crowd-sourced on Amazon Mechanical Turk. Each summary is judged by 3 annotators, and the majority vote is used as the final label.

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