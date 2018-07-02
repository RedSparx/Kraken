import numpy as np
import matplotlib.pyplot as plt

# Hypothesis 1:
P_XY_H1 = np.array([[0,0,0,1],
              [0,0,0,1],
              [0,0,0,1],
              [0,0,0,1]])*(1/4)

# Hypothesis 2:
P_XY_H2 = np.array([[0,0,0,0],
              [0,0,0,0],
              [1,1,1,1],
              [0,0,0,0]])*(1/4)

# Hypothesis 3:
P_XY_H3 = np.array([[0,1,0,0],
              [0,0,1,0],
              [0,0,0,1],
              [0,0,0,0]])

P_H1 = 0.05
P_H2 = 0.15
P_H3 = 0.8

P_XY = P_XY_H1*P_H1 + P_XY_H2*P_H2 + P_XY_H3*P_H3
print(P_XY)

plt.imshow(P_XY,cmap='coolwarm')
plt.show()