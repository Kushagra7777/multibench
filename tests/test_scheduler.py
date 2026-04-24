import torch
import torch.nn as nn

from utils.scheduler import LRCosineAnnealingScheduler, FixedScheduler


def test_fixed_scheduler_step():
    sched = FixedScheduler(lr=0.01)
    assert sched.step() == 0.01
    assert sched.step() == 0.01


def test_fixed_scheduler_update_optimizer():
    sched = FixedScheduler(lr=0.005)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sched.update_optimizer(optimizer)
    for pg in optimizer.param_groups:
        assert pg['lr'] == 0.005


def test_cosine_scheduler_step_in_range():
    sched = LRCosineAnnealingScheduler(
        eta_max=0.1, eta_min=0.001, Ti=10, Tmultiplier=2, num_batches_per_epoch=100
    )
    lr = sched.step()
    assert sched.eta_min <= lr <= sched.eta_max


def test_cosine_scheduler_starts_at_max():
    sched = LRCosineAnnealingScheduler(
        eta_max=0.1, eta_min=0.0, Ti=1, Tmultiplier=2, num_batches_per_epoch=1
    )
    lr = sched.step()
    assert lr == 0.1


def test_cosine_scheduler_resets_on_cycle_end():
    sched = LRCosineAnnealingScheduler(
        eta_max=0.1, eta_min=0.0, Ti=1, Tmultiplier=2, num_batches_per_epoch=1
    )
    sched.step()  # Tcur=0 -> eta=eta_max, no reset yet
    sched.step()  # Tcur=1 -> eta=eta_min, triggers reset
    assert sched.Tcur == 0
    assert sched.Ti == 2


def test_cosine_scheduler_update_optimizer():
    sched = LRCosineAnnealingScheduler(
        eta_max=0.05, eta_min=0.001, Ti=10, Tmultiplier=2, num_batches_per_epoch=10
    )
    sched.step()
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sched.update_optimizer(optimizer)
    for pg in optimizer.param_groups:
        assert pg['lr'] == sched.eta
