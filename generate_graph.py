import numpy as np
import time
import argparse
import os

from package.graphs import PreferentialAttachmentNetwork

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("--alpha", help="alpha parameter.", type=float)
parser.add_argument("--beta", help="beta parameter.", type=float)
parser.add_argument("--n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("--n_samples", help="Number of graphs to generate.", type=int)
parser.add_argument("--output_directory", help="Directory to save graphs in.", type=str)

args = parser.parse_args()

pan_obj = PreferentialAttachmentNetwork()

start = time.time()
pan_obj.generate_sample(alpha = args.alpha, beta =args.beta, n_nodes=args.n_nodes, disable_progress_bar=False)
end = time.time()

print(end-start)