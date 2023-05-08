"""Top level module for torchbuilder"""

__descr__ = "Simple utility to compute shape side while building a nnet"
__version__ = "0.0.1"
__license__ = "BSD 3-Clause License"
__author__ = "setepenre"
__author_email__ = "setepenre@outlook.com"
__copyright__ = "2023 setepenre"
__url__ = "https://github.com/Delaunay/torchbuilder"


import torch
import torch.nn as nn


class NNBuilder:
    """Wraps torch.nn to allow size computation"""

    def __init__(self, size, batch_size=1, pattern="CHW") -> None:
        self.batch = torch.zeros((batch_size, *size))
        self._shape = self.batch.shape
        self.pattern = pattern

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def input(self):
        return self.batch

    def shape(self):
        return self._shape

    def get_dim(self, name):
        if len(self._shape) - 1 != len(self.pattern):
            raise RuntimeError(
                f"shape ({tuple(self._shape[1:])[0]}) is not compatible with {self.pattern}"
            )

        i = self.pattern.find(name)

        if i == -1:
            raise RuntimeError(
                f"shape ({tuple(self._shape[1:])[0]}) is not compatible with {self.pattern}"
            )

        return self._shape[i + 1]

    def channel(self):
        return self.get_dim("C")

    def height(self):
        return self.get_dim("H")

    def width(self):
        return self.get_dim("W")

    def add(self, layer):
        if self.batch is None:
            return layer

        self.batch = layer(self.batch)
        self._shape = self.batch.shape

        return layer

    def wrap_layer(self, layer):
        def fun(*args, **kwargs):
            return self.add(layer(*args, **kwargs))

        return fun

    def __getattr__(self, key):
        if hasattr(nn, key):
            layer = getattr(nn, key)

            if issubclass(layer, nn.Module):
                return self.wrap_layer(layer)

            return layer

        raise AttributeError()
