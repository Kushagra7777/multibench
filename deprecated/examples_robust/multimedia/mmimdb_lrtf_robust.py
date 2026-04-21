from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader, get_dataloader_robust
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from fusions.common_fusions import LowRankTensorFusion
from training_structures.Simple_Late_Fusion import train, test
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.path.expanduser('~/multibench/MultiBench/datasets/imdb'))
sys.path.append(os.path.expanduser('~/multibench/MultiBench'))

# from get_data_robust import get_dataloader, get_dataloader_robust

traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', batch_size=128, vgg=True)
robustdata = get_dataloader_robust(
    '../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5', batch_size=128)

encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
head = Linear(512, 23).to(device)
fusion = LowRankTensorFusion([512, 512], 512, 128).to(device)


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", regularization=False,
          save=filename, optimtype=torch.optim.AdamW, lr=8e-3, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_lrtf')


def testprocess(model, testdata):
    return test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")


general_test(testprocess, filename, robustdata, multi_measure=True)
