//
//  main.m
//  Balancer
//
//  Created by Elijah Fox on 10/2/21.
//  Copyright © 2021 Elijah Fox. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <limits>
#include <vector>
#include <random>
#include <algorithm>
#include <list>
#include <functional>
#include <queue>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;
# define INF 0x3f3f3f3f

//Balancer Weighted pool formula: V = PIt Bt ^ Wt, V is a constant, B is an asset's balance and W is an assets weight in the pool
//(weights are set in the beginning and do not change -> need to optimize)

class Pool {
public:
    vector<double> reserves;
    vector<double> weights;
    vector<int> tokens;
    double totalReserves;
    
    Pool(int num_tokens, vector<double> reserves_in, vector<double> weights_in, vector<int> tokens_in) {
        totalReserves = 0;
        for (int i = 0; i < num_tokens; ++i) {
            reserves.push_back(reserves_in[i]);
            weights.push_back(weights_in[i]);
            tokens.push_back(tokens_in[i]);
            totalReserves += reserves_in[i];
        }
    }
    
    double calculate_price_slippage(uint16_t source, uint16_t target, double amount) {
        double price = (reserves[target]/ weights[target]) / (reserves[source] / weights[source]);
        double no_slippage = price * amount;
        double with_slippage = reserves[target] * (1 - (reserves[target] / pow( (reserves[target] + amount) , (weights[source]/ weights[target]) ) ) );
        return no_slippage - with_slippage;
    }
};

//Similar Pools are pools that share the same tokens, this is used for finding the routing scheme
class SimilarPools {
public:
    vector<Pool> pools;
    vector<double> split;
    double totalLiquidity;
    
    //constructor
    SimilarPools(vector<Pool> pools_in) {
        for (size_t s = 0; s < pools_in.size(); s++) {
            pools.push_back(pools_in[s]);
        }
    }
    
    //calculates total liquidity across pools
    void calculate_total_liquidity() {
        totalLiquidity = 0;
        for (Pool p: pools) {
            totalLiquidity += p.totalReserves;
        }
    }
    
    void calculate_routing_split() {
        calculate_total_liquidity();
        for (Pool p: pools) {
            double split_share = p.totalReserves / totalLiquidity;
            split.push_back(split_share);
        }
    }
};

int main(){
    vector<double> reserves = {1000, 1000};
    vector<double> reserves0 = {2000, 2000};
    vector<double> weights = {.5, .5};
    vector<int> tokens = {0, 1};
    
    int num_tokens = 2;
    Pool test0(num_tokens, reserves0, weights, tokens);
    Pool test1(num_tokens, reserves, weights, tokens);
    Pool test2(num_tokens, reserves, weights, tokens);
    
    vector<Pool> pools = {test1, test2};
    
    SimilarPools simPtest(pools);
    simPtest.calculate_routing_split();
    
    for (size_t i = 0; i < pools.size(); ++i) {
        cout << simPtest.pools[i].calculate_price_slippage(0, 1, (100 * simPtest.split[i])) << "\n";
    }
    cout << test0.calculate_price_slippage(0, 1, 100) << endl;
    
    return 1;
}
