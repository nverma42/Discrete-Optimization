#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
from collections import namedtuple
from sklearn.metrics.pairwise import euclidean_distances
import threading
import numpy as np

Point = namedtuple("Point", ['x','y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def TourLength(solution, d_mat):
    node_from = solution[-1]
    node_to = solution[0]
    obj = d_mat[node_from, node_to]
    for index in range(len(solution)-1):
        obj += d_mat[solution[index], solution[index+1]]
    return obj

def SwapNodes(curr_solution, temp_solution, d_mat, curr_obj, i, k):
    # The sequence to reverse is: curr_solution[i], curr_solution[i+1], .., curr_solution[k]
    q = i
    for p in range(k, i-1, -1):
          temp_solution[p] = curr_solution[q]
          q += 1

    # return the new tour length back to the caller. In the new tour we remove edges [i-1, i] and [k, k+1]
    # and add edges [i-1, k] and [i, k+1]
    new_tour_length1 = curr_obj
    new_tour_length1 = new_tour_length1 - d_mat[curr_solution[i-1], curr_solution[i]] + d_mat[curr_solution[i-1], curr_solution[k]]
    new_tour_length1 = new_tour_length1 - d_mat[curr_solution[k], curr_solution[k+1]] + d_mat[curr_solution[i], curr_solution[k+1]]
    
    #new_tour_length2 = TourLength(temp_solution,d_mat)
    #assert(math.fabs(new_tour_length1 - new_tour_length2) > 0.0001)
    #if (math.fabs(new_tour_length1 - new_tour_length2) > 0.0001):
    #    print("Reached here")
    return new_tour_length1

def OptimizeTour2Opt(nodeCount, best_obj, solution, d_mat, initial_T, cooling_rate):
    # Compute the starting tour length
    print('best obj:' + str(best_obj))

    # Initialize simulated annealing parameters
    T = initial_T
    cooling_rate = cooling_rate
    curr_solution = [solution[j] for j in range(nodeCount)]
    curr_obj = best_obj
    

    # Loop until system has cooled.
    iter = 0
    while T > 1:
        for i in range(nodeCount-1):
            # Look for contiguous sequence of k nodes with the largest length. Reverse such a sequence if there is improvement in objective function.
            for k in range(i+1, nodeCount-1):
                new_solution = [curr_solution[i] for i in range(nodeCount)]
                
                # Get tour length for current solution
                new_obj = SwapNodes(curr_solution, new_solution, d_mat, curr_obj, i, k)

                # decide if we should accept the new solution
                prob = random.uniform(0,1)
                if (curr_obj > new_obj or math.exp((curr_obj - new_obj)/T) > prob):
                    for j in range(nodeCount):
                        curr_solution[j] = new_solution[j]

                    curr_obj = new_obj
                    #print('curr_obj:' + str(curr_obj) + ' best obj:' + str(best_obj))

                # update the best objective value
                if (new_obj < best_obj):
                    best_obj = new_obj
                    print('iteration:' + str(iter) + ' best obj:' + str(best_obj))
                    for j in range(nodeCount):
                        solution[j] = new_solution[j]
        
        # Cooling schedule
        T *= (1-cooling_rate)
        iter += 1
    return best_obj

def SwapNodesParallel(index, solution, d_mat, best_obj, i, range_start, range_end, thread_best_objs, thread_best_solution):
    thread_best_objs[index] = best_obj
    #print("thread:" + str(index) + ": Range " + str(range_start) + "-" + str(range_end))
    for k in range(range_start, range_end, 1):
        new_solution = [solution[i] for i in range(len(solution))]
        new_obj = SwapNodes(solution, new_solution, d_mat, best_obj, i, k)
        
        if (thread_best_objs[index] > new_obj):
            thread_best_objs[index] = new_obj
            for j in range(len(solution)):
                thread_best_solution[index][j] = new_solution[j]

def OptimizeTour2OptParallel(max_iters, solution, d_mat):
    # Compute the starting tour length
    best_obj = TourLength(solution, d_mat)
    print('best obj:' + str(best_obj))
    num_nodes = len(solution)
   
    # Loop until system has cooled.
    iter = 0
    max_threads = 10
    threads = list()
    thread_best_objs = list()
    for index in range(max_threads):
        thread_best_objs.append(best_obj)
        
    thread_best_solution = list()
    for index in range(max_threads):
        temp_solution = [solution[j] for j in range(num_nodes)]
        thread_best_solution.append(temp_solution)
        
    while iter < max_iters:
        for i in range(num_nodes-1):
            threads.clear()
            chunk = int((num_nodes - i) / max_threads)
            for index in range(max_threads):
                range_start = i + index * chunk
                if (index == max_threads-1):
                    range_end = num_nodes - 1
                else:
                    range_end = range_start + chunk
                    
                t = threading.Thread(target=SwapNodesParallel, args=(index, solution, d_mat, best_obj, i, range_start, range_end, thread_best_objs, thread_best_solution,))
                threads.append(t)
                t.start()
            
            # wait for threads to finish
            for index in range(max_threads):
                threads[index].join()
                
            # update the best objective value by querying output of threads
            for index in range(max_threads):
                new_obj = thread_best_objs[index]
                if (new_obj < best_obj):
                    best_obj = new_obj
                    print('iteration:' + str(iter) + ' best obj:' + str(best_obj))
                    for j in range(num_nodes):
                        solution[j] = thread_best_solution[index][j]
                
        
        iter += 1
        
    return best_obj

def GreedyTourOpt(nodeCount, d_mat):
    next = 0
    solution = []
    for i in range(nodeCount):
        solution.append(next)
        # Find the next shortest node
        d_arr = np.argsort(d_mat[next])
        for j in range(nodeCount):
            if (not d_arr[j] in solution):
                next = d_arr[j]
                break
    best_obj = TourLength(solution, d_mat)
    
    return best_obj, solution
      
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # Create euclidean distance matrix
    d_mat = euclidean_distances(points, points)
 
    # set very high values for self distance
    for i in range(nodeCount):
        d_mat[i,i] = math.exp(100)
    
    #print(d_mat)
    best_solution = [-1 for j in range(nodeCount)]
    best_obj = math.exp(100)
    max_restarts = 10
    # solve based on parameters best for the case.
    if (nodeCount == 51 or nodeCount == 100):
        # tune SA parameters
        initial_T = 100000
        cooling_rate = 0.01
        
        for restarts in range(max_restarts):
            curr_obj, curr_solution = GreedyTourOpt(nodeCount, d_mat)
            curr_obj = OptimizeTour2Opt(nodeCount, curr_obj, curr_solution, d_mat, initial_T, cooling_rate)
            
            if (best_obj > curr_obj):
                best_obj = curr_obj
                best_solution = [curr_solution[i] for i in range(nodeCount)]
            
    elif (nodeCount == 200):
        # tune SA parameters
        initial_T = 10000
        cooling_rate = 0.1
        for restarts in range(max_restarts):
            curr_obj, curr_solution = GreedyTourOpt(nodeCount, d_mat)
            curr_obj = OptimizeTour2Opt(nodeCount, curr_obj, curr_solution, d_mat, initial_T, cooling_rate)
            if (best_obj > curr_obj):
                best_obj = curr_obj
                best_solution = [curr_solution[i] for i in range(nodeCount)]
    elif (nodeCount == 574):
        best_obj, best_solution = GreedyTourOpt(nodeCount, d_mat)
        best_obj = OptimizeTour2OptParallel(4, best_solution, d_mat)
    elif (nodeCount == 1889):
        best_obj, best_solution = GreedyTourOpt(nodeCount, d_mat)
        best_obj = OptimizeTour2OptParallel(1, best_solution, d_mat)
    else:
        best_obj, best_solution = GreedyTourOpt(nodeCount, d_mat)
    
    # prepare the solution in the specified output format
    output_data = '%.2f' % best_obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

