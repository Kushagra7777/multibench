import torch
import numpy as np

from utils.evaluation_metric import (
    multiclass_acc,
    eval_mosei_senti_return,
    eval_mosei_senti,
    eval_mosi,
)


def test_multiclass_acc_perfect():
    preds = np.array([1.0, 2.0, 3.0])
    truths = np.array([1.0, 2.0, 3.0])
    assert multiclass_acc(preds, truths) == 1.0


def test_multiclass_acc_half():
    preds = np.array([1.0, 2.0, 1.0, 2.0])
    truths = np.array([1.0, 2.0, 2.0, 1.0])
    assert multiclass_acc(preds, truths) == 0.5


def test_multiclass_acc_zero():
    preds = np.array([1.0, 1.0])
    truths = np.array([2.0, 2.0])
    assert multiclass_acc(preds, truths) == 0.0


def test_eval_mosei_senti_return_perfect():
    results = torch.tensor([1.0, -1.0, 1.0, -1.0])
    truths = torch.tensor([1.0, -1.0, 1.0, -1.0])
    mae, corr, mult_a7, f_score, acc = eval_mosei_senti_return(results, truths)
    assert mae == 0.0
    assert acc == 1.0


def test_eval_mosei_senti_return_exclude_zero():
    results = torch.tensor([1.0, -1.0, 0.5])
    truths = torch.tensor([1.0, -1.0, 0.5])
    mae, corr, mult_a7, f_score, acc = eval_mosei_senti_return(
        results, truths, exclude_zero=False
    )
    assert mae == 0.0


def test_eval_mosei_senti_prints(capsys):
    results = torch.tensor([1.0, -1.0, 1.0, -1.0])
    truths = torch.tensor([1.0, -1.0, 1.0, -1.0])
    eval_mosei_senti(results, truths)
    out = capsys.readouterr().out
    assert "MAE" in out
    assert "Accuracy" in out


def test_eval_mosi_prints(capsys):
    results = torch.tensor([1.0, -1.0])
    truths = torch.tensor([1.0, -1.0])
    eval_mosi(results, truths)
    out = capsys.readouterr().out
    assert "MAE" in out
