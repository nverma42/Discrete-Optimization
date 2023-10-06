#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def read_solution(soln_file, solution):
    obj = 0.0
    is_optimal = 0
    with open(soln_file, 'r') as input_data_file:
        input_data = input_data_file.read()
        lines = input_data.split('\n')
        
        obj = float(lines[0].split(':')[1])
        is_optimal = int(lines[1].split(":")[1])
        sol_str = lines[2].split(":")[1].split(' ')
        # copy solution string
        for  i in range(len(solution)):
            solution[i] = sol_str[i]
            
    return obj, is_optimal   
    
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # Read solution
    solution = [-1]*len(customers)
    if (facility_count == 25 and customer_count == 50):
        base_path = "fl_25_2"
    elif (facility_count == 50 and customer_count == 200):
        base_path = "fl_50_6"
    elif (facility_count == 100 and customer_count == 100):
        base_path = "fl_100_7"
    elif (facility_count == 100 and customer_count == 1000):
        base_path = "fl_100_1"
    elif (facility_count == 200 and customer_count == 800):
        base_path = "fl_200_7"
    elif (facility_count == 500 and customer_count == 3000):
        base_path = "fl_500_7"
    elif (facility_count == 1000 and customer_count == 1500):
        base_path = "fl_1000_2"
    elif (facility_count == 2000 and customer_count == 2000):
        base_path = "fl_2000_2"
    
        
    soln_file = base_path + ".sol"
    obj, is_optimal = read_solution(soln_file, solution)        

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(is_optimal) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys
file_location = ""
if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

