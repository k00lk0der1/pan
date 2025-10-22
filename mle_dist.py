import os
from package.graphs import PreferentialAttachmentNetwork
import dill

parser = argparse.ArgumentParser("PAN Random Graph Generator")

parser.add_argument("--alpha", help="alpha parameter.", type=float)
parser.add_argument("--beta", help="beta parameter.", type=float)
parser.add_argument("--n_nodes", help="Number of nodes in each graph.", type=int)
parser.add_argument("--output_directory", help="Directory to save graphs in.", type=str)


args = parser.parse_args()
filenames = os.listdir(args.output_directory)

filtered_files = [
    filename for filename in filenames if (
        f"alpha={args.alpha}" in filename and
        f"beta={args.beta}" in filename and
        f"nodes={args.n_nodes}" in filename
    )
]

n = len(filtered_files)

print("N Samples :", n)

