import numpy as np

from utils.search_tools import sample_k_configurations


def test_sample_k_configurations_all_zero_accuracies():
    np.random.seed(0)
    configurations = ["a", "b", "c"]

    samples = sample_k_configurations(configurations, [0.0, 0.0, 0.0], 2, 1.0)

    assert len(samples) == 2
    assert set(samples).issubset(set(configurations))


def test_sample_k_configurations_partial_zero_accuracies():
    np.random.seed(0)
    configurations = ["a", "b", "c"]

    samples = sample_k_configurations(configurations, [1.0, 0.0, 0.0], 2, 1.0)

    assert len(samples) == 2
    assert set(samples).issubset(set(configurations))
