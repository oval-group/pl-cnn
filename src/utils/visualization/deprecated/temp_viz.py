import numpy as np
import os
import cPickle as pickle

import seaborn as sns
sns.set(style="white", palette="Set2")

import matplotlib.pyplot as plt

log_dir = "./experiments/log/backprop/"
log_dir = "./experiments/paper/temp/"
# filename = log_dir + "01-11-2016--11-51-41-0.p"

for filename in sorted(os.listdir(log_dir)):

	filename = log_dir + filename

	if filename[-2:]!=".p" or filename.split("-")[1] != "11":
		continue

	print(filename)
	exp = pickle.load(open(filename, "rb"))

	title = "%s-C_%g-l_rate_%s" % (exp.dataset.solver, exp.dataset.C, exp.comment.split("rate: ").pop())

	# export_name = "./experiments/paper/cifar10/bp/%s" % title
	# pickle.dump(exp, open(export_name + ".p", "wb"))

	fig = plt.figure()
	fig.suptitle(title)
	ax1 = fig.add_subplot(211)

	train_err, = ax1.plot(exp.train_time, exp.train_err, label="Training error")
	val_err, = ax1.plot(exp.val_time, exp.val_err, label="Validation error")
	ax1.legend(handles=[train_err, val_err])
	ax1.set_ylabel("Error")

	ax2 = fig.add_subplot(212)
	train_acc, = ax2.plot(exp.train_time, exp.train_acc, label="Training accuracy")
	val_acc, = ax2.plot(exp.val_time, exp.val_acc, label="Validation accuracy")
	ax2.set_ylim([0., 100.])
	ax2.legend(handles=[train_acc, val_acc])
	ax2.set_xlabel("Time (s)")
	ax2.set_ylabel("Accuracy (%)")

	# plt.savefig(export_name + ".png")
	# plt.close()
	plt.show()
