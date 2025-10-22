import os
import argparse
from package.graphs import PreferentialAttachmentNetwork
import dill
import numpy as np
import pandas as pd
import multiprocess

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("--alpha", help="alpha parameter.", type=float)
parser.add_argument("--beta", help="beta parameter.", type=float)
parser.add_argument("--n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("--output_directory", help="Directory to save graphs in.", type=str)
parser.add_argument("--n_samples_max", help="Maximum number of samples to use", type=int)
parser.add_argument("--n_processes", help="Number of processes/cores to use to generate graphs", type=int)


args = parser.parse_args()
filenames = os.listdir(args.output_directory)

filtered_files = [
    os.path.join(args.output_directory, filename) for filename in filenames if (
        f"alpha={args.alpha}" in filename and
        f"beta={args.beta}" in filename and
        f"nodes={args.n_nodes}" in filename
    )
][:args.n_samples_max]

n = len(filtered_files)

print("Number of Samples :", n)

def numerical_mle(pan_obj_filename):
    return dill.load(open(pan_obj_filename, "rb")).numerical_mle().x

mp_pool = multiprocess.Pool(processes=args.n_processes)

mles = np.array(mp_pool.map(numerical_mle, filtered_files))

mp_pool.close()
mp_pool.join()


mles = pd.DataFrame(mles)

print(mles.summary())

print(mles.cov())

