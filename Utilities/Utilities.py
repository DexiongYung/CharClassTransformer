import math
import matplotlib.pyplot as plt

def plot_losses(loss: list, x_label: str, y_label: str, folder: str = "Result", filename: str = None):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'r--', label="Loss")
    plt.title("Losses")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()