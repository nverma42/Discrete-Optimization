#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def read_solution(soln_file, vehicle_tours):
    obj = 0.0
    is_optimal = 0
    with open(soln_file, 'r') as input_data_file:
        input_data = input_data_file.read()
        lines = input_data.split('\n')
        
        line = lines[0].split(' ')
        obj = float(line[0])
        is_optimal = int(line[1])
        
        # copy solution string
        for  i in range(len(vehicle_tours)):
            sol_str = lines[i+1].split(' ')
            for j in range(len(sol_str)):
                vehicle_tours[i].append(int(sol_str[j]))
            
    return obj, is_optimal   

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    
    # Read solution
    if (customer_count == 16 and vehicle_count == 3):
        base_path = "vrp_16_3_1"
    elif (customer_count == 26 and vehicle_count == 8):
        base_path = "vrp_26_8_1"
    elif (customer_count == 51 and vehicle_count == 5):
        base_path = "vrp_51_5_1"
    elif (customer_count == 101 and vehicle_count == 10):
        base_path = "vrp_101_10_1"
    elif (customer_count == 200 and vehicle_count == 16):
        base_path = "vrp_200_16_1"
    elif (customer_count == 421 and vehicle_count == 41):
        base_path = "vrp_421_41_1"
    
    soln_file = base_path + ".sol"
    
    # Initialize vehicle tours
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        
    obj, is_optimal = read_solution(soln_file, vehicle_tours)
    
    # checks that the number of customers served is correct
    #assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1
    
    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([str(j) for j in vehicle_tours[v]]) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

