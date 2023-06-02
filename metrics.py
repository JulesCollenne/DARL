import matplotlib.pyplot as plt


def plot_losses(losses, filename):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.savefig(filename)
