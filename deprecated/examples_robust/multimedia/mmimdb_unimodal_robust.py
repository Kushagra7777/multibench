from unimodals.common_models import MLP
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader, get_dataloader_robust
from training_structures.unimodal import train, test
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append(os.path.expanduser('~/multibench/MultiBench/datasets/imdb'))

traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', batch_size=128, vgg=True)
robustdata = get_dataloader_robust(
    '../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5', batch_size=128)

robustdata = robustdata[1:]
encoders = MLP(300, 512, 512).to(device)
# encoders=MLP(4096,1024,512).to(device)
head = MLP(512, 256, 23).to(device)


def trainprocess(filename_encoder, filename_head):
    train(encoders, head, traindata, validdata, 1000, early_stop=True, task="multilabel", save_encoder=filename_encoder, modalnum=0,
          save_head=filename_head, optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_unimodal', encoder=True)


def testprocess(encoder, head, testdata):
    return test(encoder, head, testdata, task="multilabel", modalnum=0)


general_test(testprocess, filename, robustdata,
             multi_measure=True, encoder=True)
