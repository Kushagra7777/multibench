import numpy as np
import torch
import torch.nn as nn

from utils.surrogate import SimpleRecurrentSurrogate, SurrogateDataloader, train_simple_surrogate


def test_surrogate_forward_shape():
    model = SimpleRecurrentSurrogate(num_hidden=4, number_input_feats=3, size_ebedding=4)
    ops = [torch.randn(2, 3) for _ in range(3)]
    out = model(ops)
    assert out.shape == (2, 1)


def test_surrogate_output_in_01():
    model = SimpleRecurrentSurrogate(num_hidden=8, number_input_feats=2, size_ebedding=8)
    ops = [torch.randn(5, 2) for _ in range(4)]
    out = model(ops)
    assert (out >= 0).all() and (out <= 1).all()


def test_surrogate_eval_model():
    model = SimpleRecurrentSurrogate(num_hidden=4, number_input_feats=3, size_ebedding=4)
    seq = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = model.eval_model(seq, torch.device('cpu'))
    assert isinstance(result, (float, np.floating))
    assert 0.0 <= result <= 1.0


def test_dataloader_add_and_get():
    loader = SurrogateDataloader()
    conf1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    conf2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    loader.add_datum(conf1, 0.8)
    loader.add_datum(conf2, 0.9)
    dataset_conf, dataset_acc = loader.get_data()
    assert len(dataset_conf) == 1
    assert dataset_conf[0].shape == (2, 2, 3)
    assert dataset_acc[0].shape == (2, 1)


def test_dataloader_deduplicates_keeps_max():
    loader = SurrogateDataloader()
    conf = np.array([[1.0, 0.0]], dtype=np.float32)
    loader.add_datum(conf, 0.5)
    loader.add_datum(conf, 0.9)
    _, dataset_acc = loader.get_data()
    assert abs(dataset_acc[0][0, 0] - 0.9) < 1e-5


def test_dataloader_to_torch():
    loader = SurrogateDataloader()
    conf = np.array([[1.0, 0.0]], dtype=np.float32)
    loader.add_datum(conf, 0.7)
    dataset_conf, dataset_acc = loader.get_data(to_torch=True)
    assert isinstance(dataset_conf[0], torch.Tensor)
    assert isinstance(dataset_acc[0], torch.Tensor)


def test_dataloader_get_k_best():
    loader = SurrogateDataloader()
    for i, acc in enumerate([0.5, 0.9, 0.7]):
        conf = np.eye(3, dtype=np.float32)[i:i+1]
        loader.add_datum(conf, acc)
    confs, accs, idx = loader.get_k_best(2)
    assert len(confs) == 2
    assert max(accs) >= 0.9


def test_train_simple_surrogate():
    model = SimpleRecurrentSurrogate(num_hidden=4, number_input_feats=2, size_ebedding=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.randn(3, 5, 2)
    outputs = torch.rand(5, 1)
    device = torch.device('cpu')
    loss = train_simple_surrogate(model, criterion, optimizer, ([inputs], [outputs]), 2, device)
    assert isinstance(loss, float)
