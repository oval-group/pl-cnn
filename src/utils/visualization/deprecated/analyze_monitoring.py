import numpy as np
import os
import cPickle as pickle

log_dir = "./experiments/log/"
filename = log_dir + "02-11-2016--11-36-17-0.p" #bestsofar
filename = log_dir + "02-11-2016--14-48-43-0.p"
filename = log_dir + "02-11-2016--22-06-46-0.p"
filename = "./experiments/paper/cifar10/svm/C_1e3_newcriterion.p"
filename = "./experiments/paper/cifar10/svm/C_1e2_new.p"
filename = "./experiments/interesting/09-11-2016--15-09-29-0.p"
# filename = log_dir + sorted(os.listdir(log_dir))[-1]

exp = pickle.load(open(filename, "rb"))

import seaborn as sns
sns.set(style="white", palette="Set1")

import matplotlib.pyplot as plt

fig = plt.figure()

taxis = np.array(exp.taxis)
taxis_val = np.array(exp.taxis)
# taxis = np.array(exp.taxis) * 1200 / exp.taxis[-1]
# taxis_val = [taxis[i] for i in range(len(taxis)) if i % 2 ==0]

sns.set(style="white", palette="Set1")
ax1 = fig.add_subplot(211)
train_err_avg, = ax1.plot(taxis, exp.perf, label="Training Objective Function")
ax1.legend(handles=[train_err_avg])
ax1.set_ylabel("Objective")

sns.set(style="white", palette="Set1")
ax2 = fig.add_subplot(212)
val_acc, = ax2.plot(taxis_val, exp.val_acc, label="Validation Accuracy")
train_acc, = ax2.plot(taxis, exp.acc, label="Training Accuracy")
# ax2.legend(handles=[train_acc, train_acc_avg])
ax2.legend(handles=[train_acc, val_acc])
ax2.set_ylabel("Accuracy (%)")

# sns.set(style="white", palette="Set1")
# ax3 = fig.add_subplot(313)
# primal, = ax3.plot(taxis, exp.primal, label="Primal")
# f, = ax3.plot(taxis, exp.f_value, label="Dual")
# ax3.legend(handles=[primal, f])
# ax3.set_xlabel("Epochs")
# ax3.set_ylabel("Objective function")

plt.show()
