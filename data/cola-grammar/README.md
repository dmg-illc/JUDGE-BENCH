# Grammatically Annotated CoLA 

Dataset introduced in the paper "Linguistic Analysis of Pretrained Sentence Encoders with Acceptability Judgments" ([Warstadt et al., arXiv 2019](https://arxiv.org/abs/1901.03438)).

The dataset consists of a grammatically annotated version of the CoLA development set. Each sentence in the CoLA development set is labeled with boolean features indicating the presence or absence of a particular grammatical construction (usually syntactic in nature). Two related sets of features are used: 63 minor features correspond to fine-grained phenomena, and 15 major features correspond to broad classes of phenomena. 

The features that are present in the converted dataset correspond to the 63 minor feautures (see the `original_data` directory)

## Convert Data
Convert the data by running:

```
python convert.py
```

The converted data will be saved as json file in the current directory.