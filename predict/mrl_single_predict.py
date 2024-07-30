# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 10:06
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

from utr.config import model_config
from utr.data.alphabet import Alphabet
from utr.model.downstream import RibosomeLoadingPredictionHead
from utr.utils.scaler import StandardScaler
lm_config='nano'
#ssmodel_weights_path = './output/ribosome_loading/mrl-epoch_ckpt-epoch=4-step=1495.ckpt'
ssmodel_weights_path = './output/mrl/checkpoints/epoch29-step3810-loss=0.074.ckpt'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class RibosomeLoadingPredictionModel(nn.Module):
    def __init__(
            self,
            lm_config: str = "nano",
            head_embed_dim: int = 32,
            head_num_blocks: int = 6,
            lr: float = 1e-3,
    ):
        super().__init__()
        #self.save_hyperparameters()

        self.scaler = StandardScaler()

        self.lm = RiNALMo(model_config(lm_config))

        self.pred_head = RibosomeLoadingPredictionHead(
            c_in=self.lm.config['model']['transformer'].embed_dim,
            embed_dim=head_embed_dim,
            num_blocks=head_num_blocks
        )

        self.loss = nn.MSELoss()

        self.lr = lr

        self.pad_idx = self.lm.config['model']['embedding'].padding_idx

    def forward(self, tokens):
        x = self.lm(tokens)["representation"]

        # Nullify padding token representations
        pad_mask = tokens.eq(self.pad_idx)
        x[pad_mask, :] = 0.0

        pred = self.pred_head(x, pad_mask)

        preds_unscaled = self.scaler.inverse_transform(pred)
        return pred,preds_unscaled

if __name__ == '__main__':

    #分词器
    tokenizer = Alphabet()

    #加载模型
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
    model.load_state_dict(adapted_state_dict,strict=True, assign=False)
    model.to(device)
    model.eval()
    print('加载模型成功')

    while True:

        seqs = input('请输入 RNA 序列：')   #AUGGCUACGUUAGCUGAACCGUAG
        #seqs = 'TCAGCGGCCATAATGTCCTCCTCTG'
        inputs = torch.tensor(tokenizer.encode(seqs), dtype=torch.int64).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred,preds_unscaled= model(inputs)
                float_value = float(preds_unscaled.item())
                #print(pred)
                print(float_value)

