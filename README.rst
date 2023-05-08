torchbuilder
=============================

|pypi| |py_versions| |codecov| |docs| |tests| |style|

.. |pypi| image:: https://img.shields.io/pypi/v/torchbuilder.svg
    :target: https://pypi.python.org/pypi/torchbuilder
    :alt: Current PyPi Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/torchbuilder.svg
    :target: https://pypi.python.org/pypi/torchbuilder
    :alt: Supported Python Versions

.. |codecov| image:: https://codecov.io/gh/Delaunay/torchbuilder/branch/master/graph/badge.svg?token=40Cr8V87HI
   :target: https://codecov.io/gh/Delaunay/torchbuilder

.. |docs| image:: https://readthedocs.org/projects/torchbuilder/badge/?version=latest
   :target:  https://torchbuilder.readthedocs.io/en/latest/?badge=latest

.. |tests| image:: https://github.com/Delaunay/torchbuilder/actions/workflows/test.yml/badge.svg?branch=master
   :target: https://github.com/Delaunay/torchbuilder/actions/workflows/test.yml

.. |style| image:: https://github.com/Delaunay/torchbuilder/actions/workflows/style.yml/badge.svg?branch=master
   :target: https://github.com/Delaunay/torchbuilder/actions/workflows/style.yml


Assuming you know the shape of the input tensors this library
allows you to compute the size of each layers.

.. code-block:: python

   from torchbuilder.core import NNBuilder

   with NNBuilder((1, 28, 28), batch_size=1, pattern='CHW') as nb:
      print(" Input Shape: ", nb.shape())
      nb.Conv2d(1, 20, 5, 1)
      print("Output Shape: ", nb.shape())

      nb.MaxPool2d(2, 2)
      print("Output Shape: ", nb.shape())
      print('Channels: ', nb.channel())



.. code-block:: python

   from torchbuilder.core import NNBuilder

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
         nb.Linear(nb.shape()[1], 500),
         nb.ReLU(),
         nb.Linear(500, num_classes),
      )


   batch = torch.rand((32, 1, 28, 28))
   model(batch)


.. code-block:: bash

   pip install torchbuilder

