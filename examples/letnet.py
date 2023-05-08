import torch
import torch.nn as nn

from torchbuilder.core import NNBuilder

num_classes = 10

with NNBuilder((1, 28, 28)) as nb:
    assert nb.channel() == 1
    assert nb.height() == 28
    assert nb.width() == 28

    model = nn.Sequential(
        nb.Conv2d(nb.channel(), 20, 5, 1),
        nb.ReLU(),
        nb.MaxPool2d(2, 2),
        nb.Conv2d(nb.channel(), 50, 5, 1),
        nb.ReLU(),
        nb.MaxPool2d(2, 2),
        nb.Flatten(),
        nb.Linear(sum(nb.shape()[1:]), 500),
        nb.ReLU(),
        nb.Linear(500, num_classes),
    )


batch = torch.rand((32, 1, 28, 28))
model(batch)
