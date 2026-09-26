import math

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


def make_etf_features(C: int = 10, d: int = 64, n_per: int = 50, noise: float = 0.01):
    etf = torch.eye(C) - torch.ones(C, C) / C
    Q = torch.linalg.qr(torch.randn(d, C))[0][:, :C].T
    means = math.sqrt(C / (C - 1)) * etf @ Q
    labels = torch.arange(C).repeat_interleave(n_per)
    feats = means[labels] + noise * torch.randn(C * n_per, d)
    return feats, labels, means
