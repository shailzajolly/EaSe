# vqa-metric-analysis
Analysis tool for evaluating VQA tasks

The script divides validation data of VQA2.0 into 2 subsets easy and difficult samples based on subjectivity of each sample. 

## Prerequisites

- python 3.6+
- numpy
- nltk(http://www.nltk.org/install.html)
- FastText Vectors (https://fasttext.cc/docs/en/english-vectors.html)

## Data

- Validation data of VQA2.0 (https://visualqa.org/download.html)

## Model used for evaluating VQA samples

- Bottom-Up and Top-Down Attention for Visual Question Answering (https://github.com/hengyuan-hu/bottom-up-attention-vqa)

## Steps to run this repository

- python main.py 
This will create two subsets of original validation question ids into easy and difficult question ids.

- python compute_accuracies.py
This will compute accuracies of the two subsets namely easy samples and hard samples. Here, the file pred_scores.json is created using model BUTD model (https://github.com/hengyuan-hu/bottom-up-attention-vqa). 

