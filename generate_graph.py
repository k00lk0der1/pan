import numpy as np
import time
import argparse
import os

from package.graphs import PreferentialAttachmentNetwork
import pymc as pm
import arviz
import multiprocess
from itertools import repeat

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("--alpha", help="alpha parameter.", type=float)
parser.add_argument("--beta", help="beta parameter.", type=float)
parser.add_argument("--n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("--n_samples", help="Number of graphs to generate.", type=int)
parser.add_argument("--output_directory", help="Directory to save graphs in.", type=str)
parser.add_argument("--random_seed", help="Random Seed for generating seeds of the graphs", type=int)
parser.add_argument("--n_processes", help="Number of processes/cores to use to generate graphs", type=int)

args = parser.parse_args()

seeds = np.random.RandomState(args.random_seed).randint(0, pow(2,32), size=args.n_samples)

def pan_objs_generating_function(pan_obj, alpha, beta, n_nodes, seed):
    pan_obj.generate_sample(
        alpha=alpha,
        beta=beta,
        n_nodes=n_nodes,
        random_state_seed=seed
    )
    return pan_obj

args_iterable = zip(
    [PreferentialAttachmentNetwork() for _ in range(args.n_samples)],
    repeat(args.alpha),
    repeat(args.beta),
    repeat(args.n_nodes),
    seeds
)

mp_pool = multiprocess.Pool(processes=args.n_processes)

pan_objs = mp_pool.starmap(pan_objs_generating_function, args_iterable)

mp_pool.close()
mp_pool.join()

for pan_obj in pan_objs:
    print(pan_obj.observed)