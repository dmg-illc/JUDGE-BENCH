# Expert-based Human Evaluations for the Submissions of WMT 2020

Dataset introduced in the paper "Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation" ([Freitag et al., TACL 2021](https://aclanthology.org/2021.tacl-1.87/)).

The dataset is a re-annotated version of the WMT English to German and Chinese to English test sets newstest2020. The annotation was carried out by raters that are professional translators and native speakers of the target language.

The datasets used for conversion use the Scalar Quality Metric (SQM) evaluation and were rated by professional translators (see the `original_data` directory)

## Convert Data
Convert the data by running:

```
python convert.py
```

The converted data will be saved as json file in the current directory.