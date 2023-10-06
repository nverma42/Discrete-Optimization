#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.setrecursionlimit(2000)
import numpy as np

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

class Solution:
    def __init__(self, room, num_items):
        self.best_integer = 0
        self.node_path = [0]*num_items
        self.room = room
        self.value = 0

# dynamic programming algorithm
def dp(items, capacity):
    
    # Construct table O[K][n], where O denotes optimal value in the knapsack
    # for capacity K and items 0..n
    (rows, cols) = (capacity+1, len(items)+1)
    O = [[0]*cols for r in range(rows)] 
    
    for n in range (len(items)+1):
        for K in range (capacity+1):
            # if capacity is 0 or number of items are 0, value is 0.
            if (K == 0 or n == 0):
                O[K][n] = 0
            else:
                item = items[n-1]
                # Decide if item n should be added to knapsack.
                if (item.weight <= K):
                    O[K][n] = max(O[K][n-1], item.value + O[K-item.weight][n-1])
                else:
                    O[K][n] = O[K][n-1]
    
    
    # determine if item is taken based on the table.
    K = capacity
    n = len(items)
    value = O[K][n]

    taken = [0]*len(items)
    while (K > 0 and n > 0):
        # Check if item n was selected
        item = items[n-1]

        if (O[K][n] > O[K][n-1]):
            taken[n-1] = 1
            # If item was taken subtract its weight
            K -= item.weight
        # Decrement item by 1
        n -= 1
    
    return value, taken

def compute_relaxation(capacity, items, binvars, start, solution):
  
    K = solution.room
    v = solution.value
    k = start + 1
    w = 0
    while (K > 0 and k < len(items)):
        if (K > items[k].weight):
            w = 1.0
        else:
            w = 1.0 * K / items[k].weight

        K -= items[k].weight * w
        v += items[k].value * w
        k += 1

    return v

def dfs(capacity, items, binvars, start, solution):
    
    # recursion base condition
    if (start > len(items)-1):
        if (solution.room >= 0 and solution.best_integer < solution.value):
            solution.best_integer = solution.value
            item_index = 0
            for i in range(len(binvars)):
                item_index = items[i].index - 1
                solution.node_path[item_index] = binvars[i]

        return

    for  b in range(1,-1,-1):
        
        # Check if problem is feasible
        if (solution.room >= items[start].weight*b):
            solution.room -= items[start].weight*b
            solution.value += items[start].value*b
            binvars[start] = b
            best_bound = compute_relaxation(capacity, items, binvars, start, solution)
        
            #recurse. as we go down the tree room vanishes and value increases. we detect infeasible conditions and return when appropriate.
            if (best_bound > solution.best_integer):
                dfs(capacity, items, binvars, start+1, solution)
            
            # Returned from recursion. so restore the room and update the value of the knapsack.
            solution.room  += items[start].weight*b
            solution.value -= items[start].value*b

    # restore binary variable while backtracking
    binvars[start] = 0

def bnb(items, capacity):

        items.sort(key=lambda x:x.value/x.weight, reverse=True)

        # Initially all the binary vars are 0
        binvars = [0]*len(items)

        solution = Solution(capacity, len(items))

        # depth first search
        dfs(capacity, items, binvars, 0, solution)

        return solution.best_integer, solution.node_path


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    #value = 0
    #weight = 0
    #taken = [0]*len(items)

    #for item in items:
    #    if weight + item.weight <= capacity:
    #        taken[item.index] = 1
    #        value += item.value
    #        weight += item.weight
    value, taken = bnb(items, capacity)
    #if (len(items) <= 200):
    #    value, taken = dp(items, capacity)
    #else:
    #    value, taken = bnb(items, capacity)
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

