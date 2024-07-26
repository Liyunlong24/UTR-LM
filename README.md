# UTR model pre-training and downstream task fine-tuning
## Introduction
The untranslated region (UTR) of RNA molecules plays a crucial role in the regulation of gene expression. Specifically, the 5' UTR located at the 5' end of the RNA molecule is a key factor in determining the efficiency of RNA translation. This project developed a semi-supervised language model of 5' UTRs that was pretrained on a combined library of random 5' UTRs and endogenous 5' UTRs from multiple species. On the basis of the pre-trained model, some downstream tasks (MRL, TE, EL, SecStructPrediction) were fine-tuned to finally achieve the prediction effect.
## Project progress

At present, the pre-training model tuning has been completed, and the fine-tuning of downstream tasks (MRL, TE, EL) has been completed and the targets have been reached.
## Installation
```bash
mamba env create -n env/environment_new.yml
```
## Pretrain
```bash
bash bash/pretrain_nano.sh
```
## Downstream
### MRL
```bash
bash bash/mrl_fine_tuning.sh
```
### TE and EL
```bash
bash bash/teel_fine_tuning.sh
```
Which Parameters you MUST to define:
* task_type: Choose from "TE" or "EL"

if you want to train TE and EL tasks on three data sets at the same time, please execute：
```bash
bash bash/teel_all_fine_tuning.sh
```

### SecStructPrediction
```bash
bash bash/ss_fine_tuning.sh
```
## Predict
```bash
python3 prdict/... .py
```
## Result
Please refer to the document for data set conditions, detailed training and fine-tuning parameters, and downstream task indicators：

[UTR-LM Project documentation](https://ab6fpiz688.feishu.cn/docx/TUGxdRBusoxg9fxHEQFczyjnnLh?from=from_copylink)

## Reference
1、A 5′ UTR language model for decoding 
untranslated regions of mRNA and 
function predictions\
(https://doi.org/10.1038/s42256-024-00823-9)

2、RiNALMo: General-Purpose RNA Language Models Can Generalize Well on
Structure Prediction Tasks\
(https://arxiv.org/abs/2403.00043)
