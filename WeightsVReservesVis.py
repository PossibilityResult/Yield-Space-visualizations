import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import random


def calculate_slippage(reserves1, reserves2, weight, amount):
    #price = (reserves2 / (1 - weight)) / (reserves1 / weight)
    with_slippage = reserves2 * (1 - ( ((reserves1) /(reserves1 + amount)) ** ((weight )/ ( 1 -  weight) ) ) ) 
    return amount - with_slippage


def calculate_cost(reserves1, reserves2, weight, amount, gas = 0):
    return calculate_slippage(reserves1, reserves2, weight, amount) + gas

num_samples = 99990

reserves_pool1 = np.array(random.sample(range(1, 99999), num_samples)) * .01
#np.linspace(10, 990, 99)

#weights1 = np.array(random.sample(range(1, 9999), num_samples)) * .0001
weights1 = np.linspace(.01, .99, 99990)

split1 = np.array(random.sample(range(1, 99999), num_samples)) * .00001
#np.linspace(.01, .99, 99)
reserves = 10000
#amounts1 = np.array(random.sample(range(1, 99999), num_samples)) * .0001
#np.linspace(.1, 9.90, 99)
amount = 100

x_data = reserves_pool1
y_data = weights1
cost_data = []

for i in range(num_samples):
    #finish cost
    cost_trade = calculate_cost(reserves_pool1[i] * weights1[i], reserves_pool1[i] * (1 - weights1[i]), weights1[i], amount)
    cost_data.append(cost_trade)

z_data = cost_data

ax = plt.axes(projection='3d')



ax.scatter(x_data, y_data, z_data, c=cost_data, cmap='gist_stern')

ax.set_xlabel("Reserves")
ax.set_ylabel("Weights Y")
ax.set_zlabel("Costs")
plt.show()
