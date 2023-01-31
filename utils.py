import numpy as np
import matplotlib.pyplot as plt

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
