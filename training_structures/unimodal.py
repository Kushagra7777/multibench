"""Implements training pipeline for unimodal comparison."""
from typing import Callable, Dict, List, Optional, Union

from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.AUPRC import AUPRC
from utils.device import get_device
from eval_scripts.performance import eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
softmax = nn.Softmax(dim=-1)


def train(
        encoder: nn.Module,
        head: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        total_epochs: int,
        early_stop: bool = False,
        optimtype: type = torch.optim.RMSprop,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        auprc: bool = False,
        save_encoder: str = 'results/models/encoder.pt',
        save_head: str = 'results/models/head.pt',
        modalnum: int = 0,
        task: str = 'classification',
        track_complexity: bool = True) -> None:
    """Train unimodal module.

    Args:
        encoder (nn.Module): Unimodal encodder for the modality
        head (nn.Module): Takes in the unimodal encoder output and produces the final prediction.
        train_dataloader (torch.utils.data.DataLoader): Training data dataloader
        valid_dataloader (torch.utils.data.DataLoader): Validation set dataloader
        total_epochs (int): Total number of epochs
        early_stop (bool, optional): Whether to apply early-stopping or not. Defaults to False.
        optimtype (torch.optim.Optimizer, optional): Type of optimizer to use. Defaults to torch.optim.RMSprop.
        lr (float, optional): Learning rate. Defaults to 0.001.
        weight_decay (float, optional): Weight decay of optimizer. Defaults to 0.0.
        criterion (nn.Module, optional): Loss module. Defaults to nn.CrossEntropyLoss().
        auprc (bool, optional): Whether to compute AUPRC score or not. Defaults to False.
        save_encoder (str, optional): Path of file to save model with best validation performance. Defaults to 'results/models/encoder.pt'.
        save_head (str, optional): Path fo file to save head with best validation performance. Defaults to 'results/models/head.pt'.
        modalnum (int, optional): Which modality to apply encoder to. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        track_complexity (bool, optional): Whether to track the model's complexity or not. Defaults to True.
    """
    def _trainprocess():
        device = get_device()
        model = nn.Sequential(encoder, head)
        op = optimtype(model.parameters(), lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                out = model(j[modalnum].float().to(device))

                if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                    loss = criterion(out, j[-1].float().to(device))
                else:
                    loss = criterion(out, j[-1].to(device))
                totalloss += loss.item() * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    out = model(j[modalnum].float().to(device))
                    if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                        loss = criterion(out, j[-1].float().to(device))
                    else:
                        loss = criterion(out, j[-1].to(device))
                    totalloss += loss.item()*len(j[-1])
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0).cpu().numpy()
            true = torch.cat(true, 0).cpu().numpy()
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy_score(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
    if track_complexity:
        all_in_one_train(_trainprocess, [encoder, head])
    else:
        _trainprocess()


def single_test(
        encoder: nn.Module,
        head: nn.Module,
        test_dataloader: DataLoader,
        auprc: bool = False,
        modalnum: int = 0,
        task: str = 'classification',
        criterion: Optional[nn.Module] = None) -> Dict[str, float]:
    """Test unimodal model on one dataloader.

    Args:
        encoder (nn.Module): Unimodal encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloader (torch.utils.data.DataLoader): Data Loader for test set.
        auprc (bool, optional): Whether to output AUPRC or not. Defaults to False.
        modalnum (int, optional): Index of modality to consider for the test with the given encoder. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.

    Returns:
        dict: Dictionary of (metric, value) relations.
    """
    model = nn.Sequential(encoder, head)
    device = get_device()
    with torch.no_grad():
        pred = []
        true = []
        totalloss = 0
        pts = []
        for j in test_dataloader:
            out = model(j[modalnum].float().to(device))
            if criterion is not None:
                loss = criterion(out, j[-1].to(device))
                totalloss += loss.item()*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if criterion is not None:
            print("loss: " + str(totalloss / totals))
        if task == "classification":
            print("acc: "+str(accuracy_score(true, pred)))
            return {'Accuracy': accuracy_score(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'F1 score (micro)': f1_score(true, pred, average="micro"), 'F1 score (macro)': f1_score(true, pred, average="macro")}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: "+str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}
        else:
            return {'MSE': totalloss / totals}


def test(
        encoder: nn.Module,
        head: nn.Module,
        test_dataloaders_all: Union[DataLoader, Dict[str, List[DataLoader]]],
        dataset: str = 'default',
        method_name: str = 'My method',
        auprc: bool = False,
        modalnum: int = 0,
        task: str = 'classification',
        criterion: Optional[nn.Module] = None,
        no_robust: bool = False) -> None:
    """Test unimodal model on all provided dataloaders.

    Args:
        encoder (nn.Module): Encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloaders_all (dict): Dictionary of noisetype, dataloader to test.
        dataset (str, optional): Dataset to test on. Defaults to 'default'.
        method_name (str, optional): Method name. Defaults to 'My method'.
        auprc (bool, optional): Whether to output AUPRC scores or not. Defaults to False.
        modalnum (int, optional): Index of modality to test on. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.
        no_robust (bool, optional): Whether to not apply robustness methods or not. Defaults to False.
    """
    if no_robust:
        def _testprocess():
            single_test(encoder, head, test_dataloaders_all,
                        auprc, modalnum, task, criterion)
        all_in_one_test(_testprocess, [encoder, head])
        return

    def _testprocess():
        single_test(encoder, head, test_dataloaders_all[list(
            test_dataloaders_all.keys())[0]][0], auprc, modalnum, task, criterion)
    all_in_one_test(_testprocess, [encoder, head])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                encoder, head, test_dataloader, auprc, modalnum, task, criterion)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)
