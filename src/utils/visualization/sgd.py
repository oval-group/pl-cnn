import seaborn as sns
import matplotlib.pyplot as plt
import cPickle as pickle


def plot(filename, export_pdf, show):

    res = pickle.load(open(filename, "rb"))

    print("C value: \t\t %g" % res['C'])
    print("Solver: \t\t %s" % res['solver'])
    print("Learning rate: \t\t %g\n" % res['learning_rate'])

    print("Training objective: \t %g" % res['train_objective'][-1])
    print("Training accuracy: \t %.2f%%\n" % res['train_accuracy'][-1])

    print("Validation objective: \t %g" % res['val_objective'][-1])
    print("Validation accuracy: \t %.2f%%\n" % res['val_accuracy'][-1])

    print("Testing objective: \t %g" % res['test_objective'])
    print("Testing accuracy: \t %.2f%%\n" % res['test_accuracy'])

    print("Train time: \t\t %g" % res['time_stamp'][-1])
    print("Number of epochs: \t %i" % len(res['val_accuracy']))

    sns_style = "white"
    sns_palette = "Set1"
    sns.set(style=sns_style, palette=sns_palette)

    fig = plt.figure()

    sns.set(style=sns_style, palette=sns_palette)
    ax1 = fig.add_subplot(211)
    train_err_avg, = ax1.plot(res['time_stamp'],
                              res['train_objective'],
                              label="Training Objective Function")
    ax1.legend(handles=[train_err_avg])
    ax1.set_ylabel("Objective")

    sns.set(style=sns_style, palette=sns_palette)
    ax2 = fig.add_subplot(212)
    val_acc, = ax2.plot(res['time_stamp'],
                        res['val_accuracy'],
                        label="Validation Accuracy")
    train_acc, = ax2.plot(res['time_stamp'],
                          res['train_accuracy'],
                          label="Training Accuracy")
    ax2.legend(handles=[train_acc, val_acc], loc=4)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Time (s)")

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
