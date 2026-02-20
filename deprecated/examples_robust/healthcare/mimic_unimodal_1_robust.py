import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
from unimodals.common_models import MLP, GRUWithLinear
from datasets.mimic.get_data_robust import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
filename_encoder = 'mimic_unimodal_1_encoder.pt'
filename_head = 'mimic_unimodal_1_head.pt'
traindata, validdata, testdata, robustdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk', tabular_robust=False)
modalnum = 1
# build encoders, head and fusion layer
#encoders = [MLP(5, 10, 10,dropout=False).to(device), GRU(12, 30,dropout=False).to(device)]
encoder = GRUWithLinear(12, 30, 15, flatten=True).to(device)
head = MLP(360, 40, 2, dropout=False).to(device)


# train
train(encoder, head, traindata, validdata, 20, auprc=False,
      modalnum=modalnum, save_encoder=filename_encoder, save_head=filename_head)

# test
encoder = torch.load(filename_encoder, weights_only=False).to(device)
head = torch.load(filename_head, weights_only=False).to(device)
acc = []
print("Robustness testing:")
for noise_level in range(len(robustdata)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(
        test(encoder, head, robustdata[noise_level], auprc=False, modalnum=modalnum))

print("Accuracy of different noise levels:", acc)
