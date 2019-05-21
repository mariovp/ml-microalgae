import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, output_file, show


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


def plot_bokeh(_real, _predicted, _model_title, _mse):
    output_file("plots/"+_model_title+".html")

    x = np.arange(0, _real.shape[0])
    _real = _real.reshape((_real.shape[0]))
    _predicted = _predicted.reshape((_predicted.shape[0]))

    title = "Real vs Predicted values (mse="+str(round(_mse, 4))+")"

    # create a new plot
    s1 = figure(width=800, plot_height=600, title=title, x_axis_label='Observation #', y_axis_label='Carbohydrates %')
    s1.line(x, _real, color="green")
    s1.circle(x, _real, legend="Real", fill_color="green", line_color="green", size=6)
    s1.line(x, _predicted, color="orange", line_dash="4 4")
    s1.triangle(x, _predicted, size=10, color="orange", alpha=0.5, legend=_model_title)

    show(s1)


def metric_to_string(metric):
    return str(round(metric, 4))
