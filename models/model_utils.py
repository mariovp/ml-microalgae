import matplotlib.pyplot as plt


def plot_model_loss(history_model):
    plt.plot(history_model['loss'], linewidth=2, label='Train')
    plt.plot(history_model['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('CNN model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()