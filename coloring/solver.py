#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from ortools.sat.python import cp_model

def color_node(node_count, node, neighbors, solution):
    # Initially all the colors are available
    available_colors = [True for j in range(node_count)]

    # Process all adjacent vertices and flag their colors as unavailable 
    if node in neighbors:
        for v in neighbors[node]:
            color = solution[v]
            if (color != -1):
                available_colors[color] = False

    # mark the solution with the first available color. This will make sure we will use minimum number of colors.
    color = -1
    for j in range(node_count):
        if (available_colors[j] == True):
            color = j
            break

    # Assign the found color
    solution[node] = color


def color_nodes_greedy(node_count, sorted_nodes, neighbors, solution):
    # greedy algorithm: 
    for item in sorted_nodes:
        node = item[0]
        # color the node from the available colors
        if (solution[node] == -1):
            color_node(node_count, node, neighbors, solution)

def color_nodes_solver(node_count, sorted_nodes, neighbors, solution):

    # Creates the model.
    model = cp_model.CpModel()

    # Create variables. Each variable denotes the color of a node whose domain is [0,..,node_count-1]
    colors = []
    for node in range(node_count):
        var = model.NewIntVar(0, node_count-1, 'color_' + str(node))
        colors.append(var)

    max_count = -1
    if (node_count == 50):
        max_count = 6
    elif (node_count == 70):
        max_count = 17
    elif (node_count == 100):
        max_count = 78
    elif (node_count == 250):
        max_count = 16
    else:
        max_count = 100

    max_color = model.NewIntVar(1, max_count-1, 'max_color')

    # Create constraints so that neighbors can't have same color.
    for i in range(node_count):
        model.Add(colors[i] <= max_color)
        if (i in neighbors):
            for v in neighbors[i]:
                if (i < v):
                    model.Add(colors[i] != colors[v])
    
    
    #symmetry breaking
    model.Add(colors[0] == 0)
    for i in range(node_count):
        model.Add(colors[i] <= i+1)

    model.Minimize(max(colors))

    # Create a solver and solves the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if (status == cp_model.OPTIMAL):
        print('objective function: %i' % solver.ObjectiveValue())
        for node in range(node_count):
            solution[node] = solver.Value(colors[node])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    node_degree = {}
    neighbors = {}
    colors = {}
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        node_from = int(parts[0])
        node_to = int(parts[1])
        edges.append((node_from, node_to))

        # populate node degree i.e. how many edges are incident upon this node.
        if node_from in node_degree:
            node_degree[node_from] += 1
        else:
            node_degree[node_from] = 1

        if node_to in node_degree:
            node_degree[node_to] += 1
        else:
            node_degree[node_to] = 1
        
        if node_from in neighbors:
            neighbors[node_from].append(node_to)
        else:
            neighbors[node_from] = [node_to]

        if node_to in neighbors:
            neighbors[node_to].append(node_from)
        else:
            neighbors[node_to] = [node_from]
  
    # sort the nodes by their degree ascending
    sorted_nodes = sorted(node_degree.items(), key=lambda x:x[1])
    
    solution = []
    for i in range(node_count):
        solution.append(-1)
        colors[i] = [j for j in range(node_count)]

    # run 100 iterations; keep the solution with the lowest optimal value
    #best_solution = node_count + 1
    #for i in range(10000):
    #    random.shuffle(sorted_nodes)
    #    temp_solution = []
    #    for k in range(node_count):
    #        temp_solution.append(-1)

    #    color_nodes_greedy(node_count, sorted_nodes, neighbors, temp_solution)
    #    if (best_solution > max(temp_solution)+1):
    #        best_solution = max(temp_solution)+1
    #        for k in range(node_count):
    #            solution[k] = temp_solution[k]


    color_nodes_solver(node_count, sorted_nodes, neighbors, solution)

    # prepare the solution in the specified output format
    output_data = str(max(solution)+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

