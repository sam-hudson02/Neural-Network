from matplotlib import pyplot as plt


def plot_net(loss: list[float], acc: list[float], val_loss: list[float],
             val_acc: list[float], title: str = 'Mnist ANN') -> None:
    fig, ax1 = plt.subplots()
    # acc and val_acc on one axis
    ax1.plot(acc, label='Training Accuracy', color='red')
    ax1.plot(val_acc, label='Validation Accuracy', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    # loss and val_loss on one axis
    ax2 = ax1.twinx()
    ax2.plot(loss, label='Training Loss', color='blue')
    ax2.plot(val_loss, label='Validation Loss', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.show()
    plt.savefig(f'./plots/{title}.png')
