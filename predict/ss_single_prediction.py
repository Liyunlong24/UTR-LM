# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 10:06
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.append(project_root)

import torch
import torch.nn as nn
from utr.model.model import RiNALMo
from utr.model.downstream import SecStructPredictionHead
from utr.config import model_config
from utr.data.alphabet import Alphabet
from utr.utils.sec_struct import prob_mat_to_sec_struct, save_to_ct
from pathlib import Path
lm_config='nano'
ssmodel_weights_path = './output/archiveII/5s/ss/v2/checkpoints/epoch=2-step=3486.ckpt'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class SecStructPredictionModel(nn.Module):
    def __init__(self, lm_config: str = "nano", num_resnet_blocks: int = 2, lr: float = 1e-5):
        super(SecStructPredictionModel, self).__init__()

        self.lm = RiNALMo(model_config(lm_config))
        self.pred_head = SecStructPredictionHead(
            self.lm.config['model']['transformer'].embed_dim,
            num_blocks=num_resnet_blocks
        )

    def forward(self, tokens):
        x = self.lm(tokens)["representation"]
        logits = self.pred_head(x[..., 1:-1, :]).squeeze(-1)
        return logits

if __name__ == '__main__':

    tokenizer = Alphabet()
    model = SecStructPredictionModel(lm_config="nano", num_resnet_blocks=2, lr=1e-5)
    checkpoint = torch.load(ssmodel_weights_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if 'threshold' in state_dict:
        threshold = state_dict['threshold']

    adapted_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            adapted_state_dict[k[6:]] = v
        else:
            adapted_state_dict[k] = v

    adapted_state_dict.pop("threshold")

    model.load_state_dict(adapted_state_dict,strict=True, assign=False)
    model.to(device)
    model.eval()
    print('加载模型成功')

    while True:

        seqs = input('请输入 RNA 序列：')   #AUGGCUACGUUAGCUGAACCGUAG
        output_file = 'output/predict_file/ss_predict.ct'

        inputs = torch.tensor(tokenizer.encode(seqs), dtype=torch.int64).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                probs = torch.sigmoid(logits)

        if probs.dtype == torch.bfloat16:
            # Cast brain floating point into floating point
            probs = probs.type(torch.float16)

        probs = probs.cpu().numpy()
        probs = probs[0]
        sec_struct_pred = prob_mat_to_sec_struct(probs=probs, seq=seqs, threshold=threshold)

        y_pred = sec_struct_pred

        Path('output/predict_file').mkdir(parents=True, exist_ok=True)

        save_to_ct(Path(output_file), sec_struct=y_pred, seq=seqs)
        print(f'{seqs}序列预测结构文件保存到{output_file}')