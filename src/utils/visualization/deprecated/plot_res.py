import cPickle as pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="Set1")


res = pickle.load(open("./experiments/log/02-11-2016--13-16-49-0.p", "rb"))
res = pickle.load(open("./experiments/paper/temp/03-11-2016--19-07-48-0.p", "rb"))
res = pickle.load(open("./experiments/paper/cifar10/bp/backup/adadelta-C_100-l_rate_0.01.p", "rb"))
# res = pickle.load(open("./experiments/log/31-10-2016--21-44-24-0.p", "rb"))

import numpy as np
print(np.mean(res.val_acc[-100:]))
print(np.std(res.val_acc[-100:]))
print(np.min(res.val_acc[-100:]))
print(np.max(res.val_acc[-100:]))

fig = plt.figure()

sns.set(style="white", palette="Set1")
ax1 = fig.add_subplot(211)


train_err, = ax1.plot(res.train_time, res.train_err, label="Training Objective Function")
ax1.set_ylim([0., 5.])
ax1.legend(handles=[train_err])
ax1.set_ylabel("Objective Function")

sns.set(style="white", palette="Set1")
ax2 = fig.add_subplot(212)
train_acc, = ax2.plot(res.train_time, res.train_acc, label="Training accuracy")
val_acc, = ax2.plot(res.val_time, res.val_acc, label="Validation accuracy")
ax2.set_ylim([0., 100.])
ax2.legend(handles=[train_acc, val_acc])
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Accuracy (%)")

plt.show()
