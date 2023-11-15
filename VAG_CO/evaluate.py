import argparse
from TestScripts import EvalConfiguration

parser = argparse.ArgumentParser()
parser.add_argument('--sample_mode', default='OG',type = str, choices = ["OG"], help='Define the sampling mode')
parser.add_argument('--Ns', default=8, type = int , help='Define the number of samples')
parser.add_argument('--batchsize', default=2,  type = int, help='Define the batch size -> the larger the faster the evaluation, Be aware of your GPU memory!')
parser.add_argument('--GPU', default=0, type = int, help='the GPU device')
args = parser.parse_args()


if(__name__ == "__main__"):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU}"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    base_path = os.getcwd()
    overall_path = base_path + "/model_checkpoints/RRG_100_MIS"
    paths = [overall_path]
    EvalConfiguration.evaluate_dataset(paths, mode = args.sample_mode, Ns = args.Ns, n_test_graphs=args.batchsize)
