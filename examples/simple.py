from torchbuilder.core import NNBuilder

with NNBuilder((1, 28, 28), batch_size=1) as nb:
    print(" Input Shape: ", nb.shape())
    nb.Conv2d(1, 20, 5, 1)
    print("Output Shape: ", nb.shape())

    nb.MaxPool2d(2, 2)
    print("Output Shape: ", nb.shape())
    print("Channels: ", nb.channel())
