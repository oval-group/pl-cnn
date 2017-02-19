import numpy as np
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

color_set = "Set1"
sns.set(style="white", palette=color_set)
colors = sns.color_palette(color_set)


def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


def plot(xp_dir, export_pdf, show):

    fig = plt.figure(figsize=(10, 3))

    matplotlib.rcParams.update({'font.size': 13})
    matplotlib.rc('xtick', labelsize=10)

    res_dict = dict()
    res_dict["train_objective"] = dict()
    res_dict["train_error"] = dict()
    res_dict["test_error"] = dict()

    solver_list = ["adagrad", "adadelta", "adam"]

    for solver in solver_list:

        base = pickle.load(open("%s%s.p" % (xp_dir, solver), "rb"))
        lwsvm = pickle.load(open("%s%s+svm.p" % (xp_dir, solver), "rb"))

        lwsvm_solver = solver + "_lwsvm"

        res_dict["train_objective"][solver] = base['train_objective'][-1]
        res_dict["train_objective"][lwsvm_solver] = \
            lwsvm['train_objective'][-1]

        res_dict["train_error"][solver] = 100. - base['train_accuracy'][-1]
        res_dict["train_error"][lwsvm_solver] = 100. - \
            lwsvm['train_accuracy'][-1]

        res_dict["test_error"][solver] = 100. - base['test_accuracy']
        res_dict["test_error"][lwsvm_solver] = 100. - lwsvm['test_accuracy']

    count = 0
    for bar_plot in ["train_objective", "train_error", "test_error"]:
        x = []
        y_base = []
        y_lwsvm = []
        for solver in solver_list:
            lwsvm_solver = solver + "_lwsvm"
            x.append(solver)
            y_base.append(res_dict[bar_plot][solver])
            y_lwsvm.append(res_dict[bar_plot][lwsvm_solver])

        ax = fig.add_subplot(131 + count)

        ind = np.arange(len(solver_list))
        width = 0.35

        rects1 = ax.bar(ind, y_base, width, color=colors[0])
        rects2 = ax.bar(ind + width, y_lwsvm, width, color=colors[1])

        ax.set_ylabel("%s %s" % tuple([str.title()
                                      for str in bar_plot.split("_")]))
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(ind + width)
        ax.set_xticklabels([solver.title() for solver in solver_list])

        # autolabel(ax, rects1)
        # autolabel(ax, rects2)

        count += 1

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    fig.legend((rects1[0], rects2[0]),
               ('Solver', 'Solver + LWSVM'),
               loc=(0.4, 0.),
               fancybox=True,
               shadow=True,
               ncol=2)

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
