# UTR model pre-training and downstream task fine-tuning
## Introduction
The untranslated region (UTR) of an RNA molecule plays a vital role in gene expression regulation. Specifically, the 5' UTR, located at the 5' end of an RNA molecule, is a critical determinant of the RNA’s translation efficiency. Language models have demonstrated their utility in predicting and optimizing the function of protein encoding sequences and genome sequences. In this study, we developed a semi-supervised language model for 5’ UTR, which is pre-trained on a combined library of random 5' UTRs and endogenous 5' UTRs from multiple species. 
## Project progress

At present, pre-training and fine-tuning for downstream tasks (MRL, TE, EL, SecStructPrediction) have been completed.
## Installation
```bash
mamba env create -n env/environment_new.yml
```
## Pretrain
```bash
bash bash/pretrain_v1.sh
```
## Downstream
### MRL
```bash
bash bash/ribosome_loading_finuetuning.sh
```
### TE and EL
```bash
bash bash/teel_finuxtuning.sh
```
Which Parameters you MUST to define:\
·task_type: Choose from "te_log" (TE task) or "rnaseq_log" (EL task)
### SecStructPrediction
```bash
bash bash/ss_fine_tuning.sh
```
## Predict
```bash
python3 prdict/... .py
```
## Reference
1、A 5′ UTR language model for decoding 
untranslated regions of mRNA and 
function predictions\
2、RiNALMo: General-Purpose RNA Language Models Can Generalize Well on
Structure Prediction Tasks
