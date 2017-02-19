import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle


def plot(filename, export_pdf, show):

    res = pickle.load(open(filename, "rb"))

    sns_style = "white"
    sns_palette = "Set1"
    sns.set(style=sns_style, palette=sns_palette)

    fig = plt.figure()

    sns.set(style=sns_style, palette=sns_palette)
    ax1 = fig.add_subplot(311)
    train_err_avg, = ax1.plot(res['time_stamp'],
                              res['train_objective'],
                              label="Training Objective Function")
    ax1.legend(handles=[train_err_avg])
    ax1.set_ylabel("Objective")

    sns.set(style=sns_style, palette=sns_palette)
    ax2 = fig.add_subplot(312)
    val_acc, = ax2.plot(res['time_stamp'],
                        res['val_accuracy'],
                        label="Validation Accuracy")
    train_acc, = ax2.plot(res['time_stamp'],
                          res['train_accuracy'],
                          label="Training Accuracy")
    ax2.legend(handles=[train_acc, val_acc], loc=4)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(30, 101)

    sns.set(style=sns_style, palette=sns_palette)
    ax3 = fig.add_subplot(313)
    primal, = ax3.plot(res['time_stamp'],
                       res['primal_objective'],
                       label="Primal")
    dual, = ax3.plot(res['time_stamp'],
                     res['dual_objective'],
                     label="Dual")
    ax3.legend(handles=[primal, dual])
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Objective function")

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
