import os
import argparse
from package.graphs import PreferentialAttachmentNetwork
import dill
import numpy as np

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("--alpha", help="alpha parameter.", type=float)
parser.add_argument("--beta", help="beta parameter.", type=float)
parser.add_argument("--n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("--output_directory", help="Directory to save graphs in.", type=str)
parser.add_argument("--n_samples_max", help="Maximum number of samples to use", type=int)


args = parser.parse_args()
filenames = os.listdir(args.output_directory)

filtered_files = [
    filename for filename in filenames if (
        f"alpha={args.alpha}" in filename and
        f"beta={args.beta}" in filename and
        f"nodes={args.n_nodes}" in filename
    )
][:args.n_samples_max]

n = len(filtered_files)

print("Number of Samples :", n)

mles = np.array(
    [
        dill.load(
            open(
                os.path.join(
                    args.output_directory,
                    filename
                )
            )
        ).numerical_mle.x for filename in filtered_files
    ]
)

print(mles.shape)

