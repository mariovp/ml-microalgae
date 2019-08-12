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
    s1 = figure(width=900, plot_height=600, title=title, x_axis_label='Observation #', y_axis_label='Carbohydrates %')
    s1.line(x, _real, color="green")
    s1.circle(x, _real, legend="Real", fill_color="green", line_color="green", size=6)
    s1.line(x, _predicted, color="orange", line_dash="4 4")
    s1.triangle(x, _predicted, size=10, color="orange", alpha=0.5, legend=_model_title)

    show(s1)


def plot_comparison(ann, cnn, lstm, knn, rf):

    output_file("plots/comparison.html")

    x_test = ann.x_test
    shape = (x_test.shape[0])
    y_real = ann.y_test.reshape(shape)

    y_predicted_ann = ann.predict(x_test).reshape(shape)
    y_predicted_cnn = cnn.predict(x_test).reshape(shape)
    y_predicted_lstm = lstm.predict(x_test).reshape(shape)
    y_predicted_knn = knn.predict(x_test).reshape(shape)
    y_predicted_rf = rf.predict(x_test).reshape(shape)

    x_range = np.arange(0, shape)

    title = "Real vs model predictions"

    # Initialize plot
    plot = figure(width=800, plot_height=600, title=title, x_axis_label='Observation #', y_axis_label='Carbohydrates %')
    # Add real values
    plot.line(x_range, y_real, color="green")
    plot.circle(x_range, y_real, legend="Real", fill_color="green", line_color="green", size=6)
    # Add ANN predictions
    # plot.line(x_range, y_predicted_ann, color="blue", line_dash="4 4")
    plot.triangle(x_range, y_predicted_ann, size=10, color="blue", alpha=0.5, legend="ANN")
    # Add CNN predictions
    # plot.line(x_range, y_predicted_cnn, color="yellow", line_dash="4 4")
    plot.square(x_range, y_predicted_cnn, size=10, color="yellow", alpha=0.5, legend="CNN")
    # Add LSTM preditions
    # plot.line(x_range, y_predicted_lstm, color="orange", line_dash="4 4")
    plot.square(x_range, y_predicted_lstm, size=10, color="orange", alpha=0.5, legend="LSTM")
    # Add KNN predictions
    # plot.line(x_range, y_predicted_knn, color="purple", line_dash="4 4")
    plot.circle(x_range, y_predicted_knn, size=10, color="purple", alpha=0.5, legend="KNN")
    # Add RF predictions
    # plot.line(x_range, y_predicted_rf, color="red", line_dash="4 4")
    plot.triangle(x_range, y_predicted_rf, size=10, color="red", alpha=0.5, legend="RF")

    show(plot)


def metric_to_string(metric):
    return str(round(metric, 4))
