import matplotlib.pyplot as plt


def plot_model_loss(model_history, model_title):
    plt.plot(model_history['loss'], linewidth=2, label='Train')
    if 'val_loss' in model_history:
        plt.plot(model_history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title(model_title+' model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


def plot_real_vs_predicted_values(_real, _predicted, _model_title, _mse):
    fig, ax = plt.subplots()

    ax.plot(_real, marker='o', ms=1.5, linestyle='-', label="Real")
    ax.plot(_predicted, marker='o', ms=1, linestyle='-', label='Predicted')

    ax.legend()
    plt.title(_model_title+" Real vs Predicted (mse="+str(round(_mse, 4))+")")
    plt.ylabel("Carbohydrates %")
    plt.xlabel("Observation #")
    plt.show()
