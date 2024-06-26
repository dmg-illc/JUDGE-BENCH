# medical-safety

Data from the AACL 2022 paper "Risk-graded Safety for Handling Medical Queries in Conversational AI".

Please cite as:

Gavin Abercrombie and Verena Rieser. 2022. Risk-graded Safety for Handling Medical Queries in Conversational AI. In *Proceedings of The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics*. Association for Computational Linguistics.

Bibtex:

        @inproceedings{abercrombie-rieser-2022-risk,
                title = {Risk-graded Safety for Handling Medical Queries in Conversational {AI}},
                author = {Gavin Abercrombie and Verena Rieser},
                booktitle = {Proceedings of The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
                year = {2022},
                publisher = {Association for Computational Linguistics}
        }

## Contents

This repo contains the following files:

- `code`
  - `medical-safety-convai.ipynb`: a Google Colab notebook
- `data`
  - `medical-safety-crowd.csv`: crowd-labelled data
  - `medical-safety-expert.csv`: expert-labelled data
  - `medical-safety-negative.csv`: (unlabelled) non-medical queries
 

## Data and labelling scheme

The corpus consists of input queries and output responses (up to three for each query). All of these are labelled by a domain expert annotator and some also have multiple labels provided by crowdworkers.

The files `medical-safety-expert.csv` and `medical-safety-negative.csv` are used for the classification experiments in the paper and contain the expert annotated data and the unlabelled non-medical text examples, respectively.

The file `medical-safety-convai.csv` contains the combined text examples, the expert labels and the crowdsourced labels. Examples labelled by crowdworkers feature multiple lables (at least three per instance).

The following column headers appear in the data files:

| Header                     | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `query`                    | Input query text                                   |
| `query-expert`             | Expert label for the query                         |
| `query-cws`                | List of crowdworker labels for the query           |
| `alexa-response`           | Alexa response                                     |
| `alexa-response-expert`    | Expert label for Alexa's response                  |
| `alexa-response-cws`       | List of crowdworker labels for Alexa's response    |
| `dialogpt-response`        | DialoGPT response                                  |
| `dialogpt-response-expert` | Expert label for DialoGPT's response               |
| `dialogpt-response-cws`    | List of crowdworker labels for DialoGPT's response |
| `reddit-response`          | Reddit (r/AskDocs) response                        |
| `reddit-response-expert`   | Expert label for response from Reddit              |
| `reddit-response-cws`      | List of crowdworker labels for the Reddit response |

Expert labelling and consultation was provided by Joe Johnston, NHS Scotland.

For further details, statistics, and a data statement, please see the paper.
