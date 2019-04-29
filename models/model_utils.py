import matplotlib.pyplot as plt


def plot_model_loss(model_history, model_title):
    plt.plot(model_history['loss'], linewidth=2, label='Train')
    plt.plot(model_history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title(model_title+' model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
