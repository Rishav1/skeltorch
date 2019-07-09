import torch.nn


class SkeletonModel(torch.nn.Module):
    """Definition of the project model.

    Defines the architecture of the neural network used in the project as a child class of `torch.nn.Module`.
    """
    param: int

    def __init__(self, param: int):
        """Constructor of the neural network.

        Args:
            param (int): hyperparameter required by the layers of the neural network.
        """
        super(SkeletonModel, self).__init__()
        self.param = param

    def forward(self, input_data):
        """Performs a forward pass through the neural network.

        Performs a forward pass throught the entire model used in the project.

        Args:
            input_data (torch.Tensor): input of the neural network.

        Returns:
            torch.Tensor: Output obtained after propagating the input through the neural network.
        """
        pass
