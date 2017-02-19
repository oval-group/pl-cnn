import numpy as np
import os
import cPickle as pickle


filename = "./experiments/paper/cifar10/svm/C_1e3_newcriterion.p"

exp = pickle.load(open(filename, "rb"))

import seaborn as sns
sns.set(style="white", palette="Set2")

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,3))

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rc('xtick', labelsize=10) 

colors = sns.color_palette("Set2")

taxis = np.array(exp.taxis) * 5500 / exp.taxis[-1]
taxis_val = [taxis[i] for i in range(len(taxis)) if i % 2 ==0]

ax1 = fig.add_subplot(211)
train_err_avg, = ax1.plot(taxis, exp.perf, label="LW-SVM", color=colors[1])
ax1.legend(handles=[train_err_avg])
ax1.set_ylabel("Training Objective Function")

ax2 = fig.add_subplot(212)
val_acc, = ax2.plot(taxis_val, exp.val_acc, label="Validation")
train_acc, = ax2.plot(taxis, exp.acc, label="Training")
ax2.legend(handles=[train_acc, val_acc])
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Accuracy (%)")

plt.show()
# plt.savefig("res_fs.pdf")
# plt.close()