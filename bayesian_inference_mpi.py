import mpi4py.MPI as MPI
import os
import argparse
from package.graphs import PreferentialAttachmentNetwork
import dill
import pymc as pm
import arviz

parser = argparse.ArgumentParser("Run bayesian inference on a PAN")

parser.add_argument("--graph_filepath", help="Path of the PAN object saved as a dill object.", type=str)

args = parser.parse_args()

if not os.path.exists(args.graph_filepath):
    raise RuntimeError(f"File {args.graph_filepath} does not exist")

# --- MPI SETUP ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# --- MPI SETUP ---

# Read pan dill obj
pan_obj = dill.load(open(args.graph_filepath, "rb"))

print(f"rank {rank} : read data")

# Run inference
idata = pan_obj.generate_posterior_samples(
    lambda : pm.Uniform('alpha', lower=-1.0, upper=10.0),
    lambda : pm.Uniform('beta', lower=0, upper=1.0),
    chains = 2,
    cores = 2,
    random_seed=42+rank
)

# Combining runs from all nodes
all_idata_list = comm.gather(idata, root=0)

# If rank is 0, 
if(rank==0):
    valid_idata = [data for data in all_idata_list if data is not None]
    merged_idata = arviz.concat(valid_idata, dim="chain")
    print(merged_idata.posterior.to_dataframe().shape)
    print(arviz.summary(merged_idata, group="posterior"))