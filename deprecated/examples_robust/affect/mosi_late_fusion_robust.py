from robustness.all_in_one import general_train, general_test
from unimodals.common_models import GRU, MLP
from get_data_robust import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append(os.path.expanduser('~/multibench/MultiBench/datasets/affect'))

# Support mosi/mosi_unaligned/mosei/mosei_unaligned/iemocap/iemocap_unaligned
traindata, validdata, robust_text, robust_vision, robust_audio, robust_timeseries = get_dataloader(
    '../../../affect/processed/mosi_data.pkl', '../../../affect/mosi', 'mosi')

# mosi
encoders = [GRU(20, 50, dropout=True, has_padding=True).to(device),
            GRU(5, 15, dropout=True, has_padding=True).to(device),
            GRU(300, 600, dropout=True, has_padding=True).to(device)]
head = MLP(665, 300, 1).to(device)
# mosei/iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).to(device), \
    GRU(74,150,dropout=True,has_padding=True).to(device),\
    GRU(300,600,dropout=True,has_padding=True).to(device)]
head=MLP(820,400,1).to(device)
'''
# iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).to(device), \
    GRU(74,150,dropout=True,has_padding=True).to(device),\
    GRU(300,600,dropout=True,has_padding=True).to(device)]
head=MLP(820,400,4).to(device)
'''
fusion = Concat().to(device)

# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 1000, True, True, task="regression", optimtype=torch.optim.AdamW,
          lr=1e-4, save=filename, weight_decay=0.01, criterion=torch.nn.L1Loss(), regularization=False)


filename = general_train(trainprocess, 'mosi_late_fusion')
# iemocap
'''
train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    optimtype=torch.optim.AdamW,lr=1e-4,save='best.pt', \
    weight_decay=0.01,regularization=False)
'''


def testprocess(model, robustdata):
    return test(model, robustdata, True, torch.nn.L1Loss(), "regression")


general_test(testprocess, filename, [
             robust_text, robust_vision, robust_audio, robust_timeseries])
# test(model,testdata,True,)
