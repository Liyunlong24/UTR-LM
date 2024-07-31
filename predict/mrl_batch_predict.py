# -*- coding: utf-8 -*-
# @Time    : 2024/7/30 14:49
import os
import sys
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.join(current_dir, '..')
# 将项目根目录添加到 PYTHONPATH 环境变量
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import torch
import torch.nn as nn
import csv
from utr.model.model import RiNALMo
from utr.config import model_config
from utr.data.alphabet import Alphabet
from utr.model.downstream import RibosomeLoadingPredictionHead
from utr.utils.scaler import StandardScaler
from Bio import SeqIO

class RibosomeLoadingPredictionModel(nn.Module):
    def __init__(
            self,
            lm_config: str = "nano",
            head_embed_dim: int = 32,
            head_num_blocks: int = 6,
    ):
        super().__init__()

        self.scaler = StandardScaler()

        self.lm = RiNALMo(model_config(lm_config))

        self.pred_head = RibosomeLoadingPredictionHead(
            c_in=self.lm.config['model']['transformer'].embed_dim,
            embed_dim=head_embed_dim,
            num_blocks=head_num_blocks
        )
        self.pad_idx = self.lm.config['model']['embedding'].padding_idx

    def forward(self, tokens):
        x = self.lm(tokens)["representation"]

        # Nullify padding token representations
        pad_mask = tokens.eq(self.pad_idx)
        x[pad_mask, :] = 0.0

        pred = self.pred_head(x, pad_mask)

        preds_unscaled = self.scaler.inverse_transform(pred)

        return pred, preds_unscaled

def load_fasta(fasta_file):
    sequences = []
    max_enc_seq_len = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_str = str(record.seq)
        sequences.append(seq_str)
        max_enc_seq_len = max(max_enc_seq_len, len(seq_str))
    return sequences, max_enc_seq_len

def save_predictions_to_csv(sequences, predictions, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Sequence', 'Prediction'])
        for seq, pred in zip(sequences, predictions):
            csvwriter.writerow([seq, pred])

lm_config = 'nano'
ssmodel_weights_path = './output/mrl/checkpoints/epoch29-step3810-loss=0.074.ckpt'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

fasta_file = './data/test.fasta'  # 修改为你的fasta文件路径
output_csv = 'predictions.csv'  # 修改为你希望保存的CSV文件路径
batch_size = 32

if __name__ == '__main__':

    sequences, max_enc_seq_len = load_fasta(fasta_file)
    predictions = []

    tokenizer = Alphabet()
    model = RibosomeLoadingPredictionModel(lm_config="nano")
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
            adapted_state_dict[k[6:]] = v  # 去掉 'model.' 前缀
        else:
            adapted_state_dict[k] = v  # 保持原始键名

    if "threshold" in adapted_state_dict:
        adapted_state_dict.pop("threshold")
    model.load_state_dict(adapted_state_dict, strict=True)
    model.to(device)
    model.eval()

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = [tokenizer.encode(seq,pad_to_len=max_enc_seq_len+2) for seq in batch_seqs]

        # Convert the list of lists to a 2D tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.int64).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred, preds_unscaled = model(inputs_tensor)

                float_values = preds_unscaled.cpu().numpy().tolist()
                predictions.extend(float_values)

    save_predictions_to_csv(sequences, predictions, output_csv)
    print('预测结果已保存到', output_csv)
