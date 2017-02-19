import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set(style="white", palette="Set2")

# plot Adjusted R2 against mu and d
fig = plt.figure(figsize=(10, 7))
ax = fig.gca(projection='3d')
x = np.arange(-5., 5., 0.3)
y = np.arange(-5., 5., 0.3)
X, Y = np.meshgrid(x, y)
Z = np.maximum(X, Y)

surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, 
        linewidth=0.5, antialiased=True)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('MaxPool(X1, X2)')
ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
ax.vieW_init(30, -100)

# fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)
# matplotlib.rcParams['lines.linewidth'] = 2
plt.tight_layout()
# plt.show()
plt.savefig("./maxpool.pdf")