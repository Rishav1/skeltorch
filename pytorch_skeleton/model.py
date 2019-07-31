import torch.nn


class SkeletonModel(torch.nn.Module):
    """Definition of the project model.

    Defines the architecture of the neural network used in the project as a child class of `torch.nn.Module`.

    Attributes:
        layer (torch.nn.Linear): main (and only) linear layer of the model.
    """
    layer: torch.nn.Linear

    def __init__(self, use_bias: bool):
        """Constructor of the model class.

        Initializes the layers required to define propagate through the model.

        Args:
            use_bias (bool): whether or not to train a bias in `self.layer`.
        """
        super(SkeletonModel, self).__init__()
        self.layer = torch.nn.Linear(1, 1, bias=use_bias)

    def forward(self, input_data: torch.Tensor):
        """Performs a forward pass through the neural network.

        Args:
            input_data (torch.Tensor): input of the neural network.

        Returns:
            output_data (torch.Tensor): output obtained after propagating the input through the neural network.
        """
        return self.layer(input_data.unsqueeze(1)).squeeze(1)
