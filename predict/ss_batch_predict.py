# -*- coding: utf-8 -*-
# @Time    : 2024/7/30 17:31
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
from utr.model.model import RiNALMo
from utr.model.downstream import SecStructPredictionHead
from utr.config import model_config
from utr.data.alphabet import Alphabet
from utr.utils.sec_struct import prob_mat_to_sec_struct, save_to_ct
from pathlib import Path
from Bio import SeqIO

lm_config='nano'
ssmodel_weights_path = './output/archiveII/5s/ss/v2/checkpoints/epoch=2-step=3486.ckpt'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class SecStructPredictionModel(nn.Module):
    def __init__(self, lm_config: str = "nano", num_resnet_blocks: int = 2, lr: float = 1e-5):
        super(SecStructPredictionModel, self).__init__()
        # 假设 RiNALMo 和 model_config 已经定义并且可以使用
        self.lm = RiNALMo(model_config(lm_config))
        self.pred_head = SecStructPredictionHead(
            self.lm.config['model']['transformer'].embed_dim,
            num_blocks=num_resnet_blocks
        )
        self.loss = nn.BCEWithLogitsLoss()

        # 初始化优化器，这里使用 Adam 作为示例
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, tokens):
        x = self.lm(tokens)["representation"]
        logits = self.pred_head(x[..., 1:-1, :]).squeeze(-1)
        return logits

def load_fasta(fasta_file):
    sequences = []
    max_enc_seq_len = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_str = str(record.seq)
        sequences.append(seq_str)

        # 更新最大长度
        max_enc_seq_len = max(max_enc_seq_len, len(seq_str))

    return sequences, max_enc_seq_len


fasta_file = './data/test.fasta'  # 修改为你的fasta文件路径
#output_csv = 'predictions.csv'  # 修改为你希望保存的CSV文件路径
batch_size = 32  # 设定batch size

if __name__ == '__main__':

    predictions = []

    sequences, max_enc_seq_len = load_fasta(fasta_file)
    print(max_enc_seq_len)

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
            adapted_state_dict[k[6:]] = v  # 去掉 'model.' 前缀
        else:
            adapted_state_dict[k] = v  # 保持原始键名

    adapted_state_dict.pop("threshold")

    model.load_state_dict(adapted_state_dict,strict=True, assign=False)
    model.to(device)
    model.eval()
    print('加载模型成功')

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = [tokenizer.encode(seq, pad_to_len=max_enc_seq_len + 2) for seq in batch_seqs]

        # Convert the list of lists to a 2D tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.int64).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(inputs_tensor)
                probs = torch.sigmoid(logits)

        if probs.dtype == torch.bfloat16:
            # Cast brain floating point into floating point
            probs = probs.type(torch.float16)

        probs = probs.cpu().numpy()

        Path('output').mkdir(parents=True, exist_ok=True)

        for i, seq in enumerate(sequences):
            prob = probs[i]
            sec_struct_pred = prob_mat_to_sec_struct(probs=prob, seq=seq, threshold=threshold)

            output_file = f'output/ss_predict_{i}.ct'
            save_to_ct(Path(output_file), sec_struct=sec_struct_pred, seq=seq)
            print(f'{seq} 序列预测结构文件保存到 {output_file}')