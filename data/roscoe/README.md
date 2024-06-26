# ROSCOE Datasets

Human judged datasets from "ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning" ([Golovneva et al., ICLR 2023](https://doi.org/10.48550/arXiv.2212.07919)).

Human annotations are collected for a subset of the following datasets:
- [CosmosQA](https://doi.org/10.18653/v1/D19-1243)
- [DROP](https://doi.org/10.18653/v1/N19-1246)
- [ESNLI](https://doi.org/10.48550/arXiv.1812.01193)
- [GSM8K](https://doi.org/10.48550/arXiv.2110.14168)

Annotations judge the reasoning quality of GPT-3's output. For each dataset, two data splits are extracted: one that judges the overall quality of the generated response, and another one that judges the quality of the individual reasoning steps. Splits are named respectively.

## Download Original Data
To convert the data, please first download the original data by running:
```
bash download_original_data.sh
```

## Convert Data
To convert the data into the data split that judges the overall quality of the generated response, run:
```
python convert_overall.py
```

For a conversion into the data split that judges the quality of the individual reasoning steps, run:
```
python convert_stepwise.py
```

The respective data will be saved as json files in the current directory.

