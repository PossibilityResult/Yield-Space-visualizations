import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

n = 120
t_arr = np.linspace(0, .95, n)
k = 1


#k ** (1 / ( 1 - temp_t ) )


def g(t, k):
    return np.linspace(0, k ** (1 / ( 1 - t )), n)

def calc_y(x, t, k):
    return ( k - x ** ( 1 - t ) ) ** ( 1 / ( 1 - t ) )

def calc_delta_y(y, x, delta_x, t, k):
    return y - ( k - ( x + delta_x ) ** ( 1 - t ) ) ** ( 1 / ( 1 - t ) )

def f(x, t, k):
    y = calc_y(0, t, k)
    return y - calc_delta_y(y, 0, x, t, k)

def gen_data(t_arr):
    full_arr = []
    for t in t_arr:
        x_arr = g(t, k)
        for x in x_arr:
            y = f(x, t, k)
            max = k ** (1 / ( 1 - t ) )
            if (.001 < y < max and t > .01):
                full_arr.append((t, x, y))
    return full_arr


data = gen_data(t_arr)
data_matrix = np.asmatrix(data)
print(data_matrix.shape)

T_data = data_matrix[:, 0]
X_data = data_matrix[:, 1]
Y_data = data_matrix[:, 2]

ax = plt.axes(projection='3d')

ax.scatter(T_data, X_data, Y_data, c=T_data, cmap='nipy_spectral')

ax.set_ylabel("Reserves X")
ax.set_zlabel("Reserves Y")
ax.set_xlabel("Time T (0-1)")

plt.show()
plt.savefig('python/yield_space_original.png')
