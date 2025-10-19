import numpy as np
import argparse

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("alpha", help="alpha parameter.", type=float)
parser.add_argument("beta", help="beta parameter.", type=float)
parser.add_argument("n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("n_samples", help="Number of graphs to generate.", type=int)

args = parser.parse_args()

print(args)
