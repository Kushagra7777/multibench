from fusions.common_fusions import Concat, ConcatWithLinear
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from training_structures.Supervised_Learning import single_test, train

class UnimodalDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return [self.x[index], self.y[index]]

    def __len__(self):
        return len(self.y)

class MultimodalDataset(Dataset):
    def __init__(self, xs, y):
        super().__init__()
        self.xs = xs
        self.y = y

    def __getitem__(self, index):
        return [*[x[index] for x in self.xs], self.y[index]]

    def __len__(self):
        return len(self.y)


class DtypeCheckingModel(nn.Module):
    def __init__(self, expected_dtype):
        super().__init__()
        self.expected_dtype = expected_dtype
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        assert inputs[0].dtype == self.expected_dtype
        batch_size = inputs[0].shape[0]
        logits = torch.zeros(batch_size, 2, device=inputs[0].device)
        logits[:, 0] = 1
        return logits

def test_Supervised_Learning_unimodal_classification():
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.long)

    torch.manual_seed(42)
    train_ds = UnimodalDataset(x, y)
    train_loader = DataLoader(train_ds)

    train([model], Concat(), nn.Identity(), train_loader, train_loader, 80)

    device = next(model.parameters()).device
    for i, o in zip(x, y):
        assert torch.allclose(torch.argmax(model(i[None].to(device)), dim=-1), o[None].to(device))

def test_Supervised_Learning_multimodal_classification():
    encoders = [nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    ) for _ in range(2)]
    fusion = ConcatWithLinear(8, 2)

    x1 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    x2 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float)
    y = torch.tensor([[1], [0], [0], [1]], dtype=torch.long)

    torch.manual_seed(42)
    train_ds = MultimodalDataset([x1, x2], y)
    train_loader = DataLoader(train_ds)

    train(encoders, fusion, nn.Identity(), train_loader, train_loader, 80)

    device = next(encoders[0].parameters()).device
    for i1, i2, o in zip(x1, x2, y):
        i = [i1.to(device), i2.to(device)]
        output = []
        for j in range(len(i)):
            output.append(encoders[j](i[j][None]))
        output = fusion(output)
        assert torch.allclose(torch.argmax(output, dim=-1), o[None].to(device))

def test_Supervised_Learning_unimodal_regression():
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

    torch.manual_seed(42)
    train_ds = UnimodalDataset(x, y)
    train_loader = DataLoader(train_ds)

    train([model], Concat(), nn.Identity(), train_loader, train_loader, 160, task='regression', objective=nn.MSELoss())

    device = next(model.parameters()).device
    for i, o in zip(x, y):
        assert torch.allclose(model(i.to(device)), o.to(device), atol=0.1)

def test_Supervised_Learning_multimodal_regression():
    encoders = [nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    ) for _ in range(2)]
    fusion = ConcatWithLinear(8, 1)

    x1 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    x2 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float)
    y = torch.tensor([[1], [0], [0], [1]], dtype=torch.float)

    torch.manual_seed(42)
    train_ds = MultimodalDataset([x1, x2], y)
    train_loader = DataLoader(train_ds)

    train(encoders, fusion, nn.Identity(), train_loader, train_loader, 80, task='regression', objective=nn.MSELoss())

    device = next(encoders[0].parameters()).device
    for i1, i2, o in zip(x1, x2, y):
        i = [i1.to(device), i2.to(device)]
        output = []
        for j in range(len(i)):
            output.append(encoders[j](i[j][None]))
        output = fusion(output)
        assert torch.allclose(output, o[None].to(device), atol=0.1)


def test_single_test_respects_input_to_float_false():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    y = torch.tensor([0, 0], dtype=torch.long)
    test_loader = DataLoader(UnimodalDataset(x, y), batch_size=2)
    model = DtypeCheckingModel(torch.long)

    result = single_test(model, test_loader, input_to_float=False)

    assert result["Accuracy"] == 1.0
