# UTR model pre-training and downstream task fine-tuning
## Introduction
The untranslated region (UTR) of RNA molecules plays a crucial role in the regulation of gene expression. Specifically, the 5' UTR located at the 5' end of the RNA molecule is a key factor in determining the efficiency of RNA translation. This project developed a semi-supervised language model of 5' UTRs that was pretrained on a combined library of random 5' UTRs and endogenous 5' UTRs from multiple species. On the basis of the pre-trained model, some downstream tasks (MRL, TE, EL, SecStructPrediction) were fine-tuned to finally achieve the prediction effect.
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
Which Parameters you MUST to define:
* task_type: Choose from "te_log" (TE task) or "rnaseq_log" (EL task)
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
