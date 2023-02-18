import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def plot_losses(args, train_loss, val_loss, model_n):
    epochs = np.arange(1, len(train_loss)+1)

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Add in a title and axes labels
    plt.title(f'{model_n}: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    #plt.xticks(np.arange(0, len(train_loss)+1, 2))

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(os.path.join(args.losses_path, f'{model_n}_losses_img.jpg'))
    plt.close()

#From the original implementation
class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size):
        """
        Same as AdaptiveAvgPool2d, with saving the shape for matrix upscaling.
        Args:
            output_size: Adaptive size of the output.
        """
        super().__init__(output_size)
        self.shape = None

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]
        return super().forward(in_tensor)

#From the original implementation
class FinalLayer(nn.Module):

    def __init__(self, norm=1, bias=-5):
        """
        Used to add a bias and a temperature scaling to the final output of a particular model.
        Args:
            norm: inverse temperature, i.e., T^{-1}
            bias: constant shift in logit space for all classes.
        """
        super().__init__()
        assert norm != 0, "Norm 0 means average pooling in the last layer of the old trainer. " \
                          "Please add size.prod() of final layer as img_size_norm to exp_params."
        self.norm = norm
        self.bias = bias

    def forward(self, in_tensor):
        out = (in_tensor.view(*in_tensor.shape[:2])/self.norm + self.bias)
        return out

#From the original implementation
class AddInverse(nn.Module):

    def __init__(self, dim=1):
        """
            Adds (1-in_tensor) as additional channels to its input via torch.cat().
            Can be used for images to give all spatial locations the same sum over the channels to reduce color bias.
        """
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        out = torch.cat([in_tensor, 1-in_tensor], self.dim)
        return out
