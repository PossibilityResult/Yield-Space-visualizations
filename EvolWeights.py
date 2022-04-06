import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np, numpy.random
from numpy.core.shape_base import _stack_dispatcher
import random

def calculate_slippage(reserves1, reserves2, weight1, weight2, amount):
    #price = (reserves2 / (1 - weight)) / (reserves1 / weight)
    with_slippage = reserves2 * (1 - ( ((reserves1) /(reserves1 + amount)) ** (weight1 / weight2 ) ) ) 
    return amount - with_slippage

def calculate_cost(reserves1, reserves2, weight1, weight2, amount, gas = 0):
    return calculate_slippage(reserves1, reserves2, weight1, weight2, amount) + gas

def random_order_book_gen(num_orders, num_tokens, max_amount):
    order_book = []
    for i in range(num_orders):
        order = random.sample(range(0, num_tokens))
        order.append(random.randint(0, max_amount))
        order_book.append(order)
    return order_book

def random_population_gen(pop_size, num_tokens):
    weight_data = []
    for i in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_tokens),size=1)[0]
        weight_data.append(weights)
    return weight_data

def loss_function(order_book, weights, reserves):
    loss = 0
    for i in range(len(order_book)):
        loss += calculate_cost( weights[order_book[i][0]] * reserves, weights[order_book[i][1]] * reserves, weights[order_book[i][0]], weights[order_book[i][1]], order_book[i][2])
    return loss

def score_population(order_book, weight_population, reserves):
    for i in range(len(weight_population)):
        weight_population[i].append(loss_function(order_book, weight_population[i], reserves))
    weight_population.sort(key=lambda x: x[3])
    return weight_population

def mutate(weights):
    num_mutes = random.randint(0, 3)
    for i in range(num_mutes):
        wd = random.randint(0,3)
        wu = random.randint(0,3)
        change = random.uniform(0, .1)
        if (weights[wu] + change >= 1 or weights[wd] <= 0):
            None
        else:
            weights[wu] += change
            weights[wd] -= change
    return weights


        



pop_size = 100
num_orders = 1000
num_tokens = 3
max_amount = 10
num_gens = 100
reserves = 1000

order_book = random_order_book_gen(num_orders, num_tokens, max_amount)

weight_population = random_population_gen(pop_size, num_tokens)

for i in range(num_gens):
    #finish cost
    
