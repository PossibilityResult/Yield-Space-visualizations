import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import random

#NEED TO FIX WEIGHTS: NOT MEASURE ACCURATE THING


def calculate_slippage(reserves1, reserves2, weight, amount):
    price = (reserves2 ** (1 - weight)) / (reserves1 ** weight)
    no_slippage = price * amount
    with_slippage = reserves2 * (1 - ( (reserves1 /(reserves1 + amount)) ** (weight / (1 - weight) ) ) ) 
    return no_slippage - with_slippage

def calculate_cost(reserves1, reserves2, weight, amount, gas = 20):
    return calculate_slippage(reserves1, reserves2, weight, amount) + gas

num_samples = 99990

reserves_pool1 = np.array(random.sample(range(1, 99999), num_samples)) * .01
#np.linspace(10, 990, 99)

weights1 = np.array(random.sample(range(1, 99999), num_samples)) * .00001
#np.linspace(.01, .99, 99)

split1 = np.array(random.sample(range(1, 99999), num_samples)) * .00001
#np.linspace(.01, .99, 99)

#amounts1 = np.array(random.sample(range(1, 99999), num_samples)) * .0001
#np.linspace(.1, 9.90, 99)
amount = 10

x_data = reserves_pool1
y_data = weights1
z_data = split1
cost_data = []

for i in range(num_samples):
    #finish cost
    cost_trade_1 = calculate_cost(reserves_pool1[i]/2, reserves_pool1[i]/2, weights1[i], split1[i] * amount)
    cost_trade_2 = calculate_cost((1000 - reserves_pool1[i]/2), (1000 - reserves_pool1[i]/2), 1 - weights1[i], (1 - split1[i]) * amount)
    total_cost = cost_trade_1 + cost_trade_2
    cost_data.append(total_cost)

ax = plt.axes(projection='3d')



ax.scatter(x_data, y_data, z_data, c=cost_data, cmap='flag')

ax.set_xlabel("Reserves ( 1 , T - 2 )")
ax.set_ylabel("Weights ( 1 , T - 2 )")
ax.set_zlabel("Split Ratio ( 1 , T - 2 ) ")
plt.show()
