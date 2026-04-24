import numpy as np
import torch
import torch.nn as nn

from eval_scripts.complexity import getallparams, all_in_one_train, all_in_one_test
from eval_scripts.performance import ptsort, AUPRC, f1_score, accuracy, eval_affect


def test_getallparams_single():
    model = nn.Linear(10, 5)
    assert getallparams([model]) == 10 * 5 + 5


def test_getallparams_multiple():
    m1 = nn.Linear(10, 5)
    m2 = nn.Linear(5, 2)
    assert getallparams([m1, m2]) == (10 * 5 + 5) + (5 * 2 + 2)


def test_all_in_one_train_calls_process(capsys):
    called = []
    all_in_one_train(lambda: called.append(1), [nn.Linear(2, 1)])
    assert len(called) == 1
    assert "Training Time" in capsys.readouterr().out


def test_all_in_one_test_calls_process(capsys):
    called = []
    all_in_one_test(lambda: called.append(1), [nn.Linear(2, 1)])
    assert len(called) == 1
    assert "Inference Time" in capsys.readouterr().out


def test_ptsort():
    assert ptsort((0.9, 1)) == 0.9
    assert ptsort((0.0, 0)) == 0.0


def test_AUPRC_perfect():
    pts = [(0.9, 1), (0.8, 1), (0.1, 0), (0.2, 0)]
    result = AUPRC(pts)
    assert 0.0 <= result <= 1.0


def test_AUPRC_all_correct():
    pts = [(1.0, 1), (0.0, 0)]
    assert AUPRC(pts) == 1.0


def test_f1_score_perfect():
    truth = torch.tensor([1, 0, 1, 0])
    pred = torch.tensor([1, 0, 1, 0])
    assert f1_score(truth, pred, average='macro') == 1.0


def test_f1_score_binary():
    truth = torch.tensor([1, 1, 0, 0])
    pred = torch.tensor([1, 0, 0, 0])
    result = f1_score(truth, pred, average='binary')
    assert 0.0 <= result <= 1.0


def test_accuracy_perfect():
    truth = torch.tensor([1, 0, 1, 0])
    pred = torch.tensor([1, 0, 1, 0])
    assert accuracy(truth, pred) == 1.0


def test_accuracy_partial():
    truth = torch.tensor([1, 0, 1, 0])
    pred = torch.tensor([1, 0, 0, 1])
    assert accuracy(truth, pred) == 0.5


def test_eval_affect_numpy_perfect():
    truths = np.array([1.0, -1.0, 1.0, -1.0])
    results = np.array([1.0, -1.0, 1.0, -1.0])
    assert eval_affect(truths, results) == 1.0


def test_eval_affect_tensor():
    truths = torch.tensor([1.0, -1.0, 1.0, -1.0])
    results = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert eval_affect(truths, results) == 1.0


def test_eval_affect_exclude_zero():
    truths = np.array([1.0, 0.0, -1.0])
    results = np.array([1.0, 0.5, -1.0])
    result_excl = eval_affect(truths, results, exclude_zero=True)
    result_incl = eval_affect(truths, results, exclude_zero=False)
    assert 0.0 <= result_excl <= 1.0
    assert 0.0 <= result_incl <= 1.0
