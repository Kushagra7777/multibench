from unimodals.common_models import MLP, VGG16, MaxOut_MLP, Linear
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader, get_dataloader_robust
from fusions.common_fusions import Concat
from training_structures.Contrastive_Learning import train, test
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')

traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', batch_size=128, vgg=True)
robustdata = get_dataloader_robust(
    '../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5', batch_size=128)

encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head = Linear(1024, 23).to(device)
refiner = MLP(1024, 3072, 4396).to(device)
#refiner = MLP(1024,2048,1024).to(device)
fusion = Concat().to(device)


def trainprocess(filename):
    train(encoders, fusion, head, refiner, traindata, validdata, 1000, early_stop=True, task="multilabel",
          save=filename, optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_contrast')


def testprocess(model, robustdata):
    return test(model, robustdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")


general_test(testprocess, filename, robustdata, multi_measure=True)
