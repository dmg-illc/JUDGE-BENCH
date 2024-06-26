# The Corpus of Linguistic Acceptability (CoLA) 

Dataset introduced in the paper "Neural Network Acceptability Judgments" ([Warstadt et al., TACL 2019](https://aclanthology.org/Q19-1040/)).

The Corpus of Linguistic Acceptability (CoLA) in its full form consists of 10657 sentences from 23 linguistics publications, expertly annotated for acceptability (grammaticality) by their original authors.

The datasets used for conversion belong to the development set of the original CoLA dataset (see the `original_data` directory)

## Convert Data
Convert the data by running:

```
python convert.py
```

The converted data will be saved as json file in the current directory.