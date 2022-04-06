import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np, numpy.random
import random

def calculate_slippage(reserves1, reserves2, weight1, weight2, amount):
    #price = (reserves2 / (1 - weight)) / (reserves1 / weight)
    with_slippage = reserves2 * (1 - ( ((reserves1) /(reserves1 + amount)) ** (weight1 / weight2 ) ) ) 
    return amount - with_slippage


def calculate_cost(reserves1, reserves2, weight1, weight2, amount, gas = 0):
    return calculate_slippage(reserves1, reserves2, weight1, weight2, amount) + gas

num_samples = 99990

x_data = []
y_data = []
z_data = []

for i in range(num_samples):
    weights = np.random.dirichlet(np.ones(3),size=1)[0]
    x_data.append(weights[0])
    y_data.append(weights[1])
    z_data.append(weights[2])

cost_data = []
reserves = 1000
amount = 100
for i in range(num_samples):
    #finish cost
    cost_trade = calculate_cost(reserves * x_data[i], reserves * y_data[i], x_data[i], y_data[i], amount)
    cost_trade += calculate_cost(reserves * z_data[i], reserves * y_data[i], z_data[i], y_data[i], 15* amount)
    cost_trade += calculate_cost(reserves * x_data[i], reserves * z_data[i], x_data[i], z_data[i], 15* amount)
    cost_data.append(cost_trade)

#z_data = cost_data

ax = plt.axes(projection='3d')



ax.scatter(x_data, y_data, z_data, c=cost_data, cmap='nipy_spectral')

ax.set_xlabel("Weights X")
ax.set_ylabel("Weights Y")
ax.set_zlabel("Weights Z")

plt.show()
