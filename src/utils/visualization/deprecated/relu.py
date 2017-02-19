import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", palette="Set2")

x = np.arange(-5, 5, 0.01)
y = 0.5 * (x + np.abs(x))

# plt.suptitle("Rectified Linear Unit")
# plt.plot(x, y)
# plt.xlim(-5.5, 5.5)
# plt.ylim(-5.5, 5.5)
# plt.show()

fig, ax = plt.subplots()

ax.set_aspect('equal')
# ax.grid(True, which='both')
ax.spines['left'].set_position('center')
ax.spines['left'].set_color('grey')
ax.spines['left'].set_alpha(0.3)
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['bottom'].set_color('grey')
ax.spines['bottom'].set_alpha(0.3)
ax.spines['top'].set_color('none')
# ax.spines['left'].set_smart_bounds(True)
# ax.spines['bottom'].set_smart_bounds(True)
# sns.despine(ax=ax, offset=0)
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.set_xlabel("x")
ax.xaxis.set_label_coords(1.05, 0.45)
ax.set_ylabel("ReLU(x)")
ax.yaxis.set_label_coords(0.45, 0.85)
ax.plot(x, y, 'b')
plt.suptitle("Rectified Linear Unit")
plt.tight_layout()
# plt.show()
plt.savefig("./relu.pdf")

