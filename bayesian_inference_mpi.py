import mpi4py.MPI as MPI
import os
import argparse
from package.graphs import PreferentialAttachmentNetwork
import dill
import pymc as pm
import arviz
import jax

parser = argparse.ArgumentParser("Run bayesian inference on a PAN")

parser.add_argument("--graph_filepath", help="Path of the PAN object saved as a dill object.", type=str)
parser.add_argument('--profile', action=argparse.BooleanOptionalAction)
parser.add_argument('--profile_dir', type=str, required=False)


args = parser.parse_args()

if not os.path.exists(args.graph_filepath):
    raise RuntimeError(f"File {args.graph_filepath} does not exist")

# --- MPI SETUP ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = os.popen('hostname').read()
# --- MPI SETUP ---

print(f"Rank : {rank}, Hostname : {hostname}")

# Read pan dill obj
pan_obj = dill.load(open(args.graph_filepath, "rb"))

inf_start_time = MPI.Wtime()
# Run inference

if(args.profile):
    if(args.profile_dir is None):
        raise ValueError("If profiling is enabled, a directory path to write profiling data to is required.")
    if(not os.path.exists(args.profile_dir)):
        raise FileNotFoundError(f"{args.profile_dir} doesn't exist")
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    jax.profiler.start_trace(
        os.path.join(
            args.profile_dir,
            str(rank)
        ),
        profiler_options=options
    )

idata = pan_obj.generate_posterior_samples(
    lambda : pm.Uniform('alpha', lower=-1.0, upper=10.0),
    lambda : pm.Uniform('beta', lower=0, upper=1.0),
    chains = 2,
    random_seed=42+rank
)

if(args.profile):
    jax.block_until_ready()
    jax.profiler.stop_trace()
    

inf_end_time = MPI.Wtime()

inf_time = inf_end_time - inf_start_time

# Combining runs from all nodes
all_idata_list = comm.gather(idata, root=0)
all_inf_time = comm.gather((rank, inf_time), root=0)

# If rank is 0, 
if(rank==0):
    all_inf_time = dict(all_inf_time)
    valid_idata = [data for data in all_idata_list if data is not None]
    merged_idata = arviz.concat(valid_idata, dim="chain")
    merged_idata.to_netcdf(args.graph_filepath+"_posterior")

    for rank in all_inf_time.keys():
        print(rank, all_inf_time[rank])

    print(max(all_inf_time.values())/min(all_inf_time.values()))
    

    
