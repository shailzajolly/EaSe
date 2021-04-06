# Vqa-metric-analysis
Analysis tool for evaluating VQA Datasets

The repository contains code of our accepted NAACL paper titled "EaSe: A Diagnostic Tool for VQA based on Answer Diversity".

## Prerequisites

- python 3.6+
- numpy
- nltk(http://www.nltk.org/install.html)
- FastText Vectors (https://fasttext.cc/docs/en/english-vectors.html)

## Data

- Validation data of VQA2.0 (https://visualqa.org/download.html)

## Model used for evaluating VQA samples

- Bottom-Up and Top-Down Attention for Visual Question Answering (https://github.com/hengyuan-hu/bottom-up-attention-vqa)
- LXMERT (https://github.com/airsplay/lxmert)

## Steps to run this repository

- python main.py 

This will create subsets of original validation question ids into easy and difficult question ids.

- python compute_accuracies.py

This will compute accuracies of the subsets. Here, the file prediction file is created using VQA models.

