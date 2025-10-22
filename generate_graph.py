import numpy as np
import time
import argparse
import os

from package.graphs import PreferentialAttachmentNetwork
import pymc as pm
import arviz

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("--alpha", help="alpha parameter.", type=float)
parser.add_argument("--beta", help="beta parameter.", type=float)
parser.add_argument("--n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("--n_samples", help="Number of graphs to generate.", type=int)
parser.add_argument("--output_directory", help="Directory to save graphs in.", type=str)
parser.add_argument("--random_seed", help="Random Seed for generating seeds of the graphs", type=int)
parser.add_argument("--n_processes", help="Number of processes/cores to use to generate graphs", type=int)

args = parser.parse_args()

#pan_objs = [PreferentialAttachmentNetwork() for _ in range(args.n_samples)]

seeds = np.random.RandomState(args.random_seed).randint(0, pow(2,32), size=args.n_samples)

print(seeds.shape)
print(seeds)