import os
import sys
import argparse
import numpy as np
import theano

from neuralnet import NeuralNet
from config import Configuration as Cfg

# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "cifar100"])
parser.add_argument("--batch_size",
                    help="batch size",
                    type=int, default=200)
parser.add_argument("--device",
                    help="Computation device to use for experiment",
                    type=str, default="cpu")
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--in_name",
                    help="name for inputs of experiment",
                    type=str)
parser.add_argument("--out_name",
                    help="name for outputs of experiment",
                    type=str, default="")
parser.add_argument("--C",
                    help="regularization hyper-parameter",
                    type=float, default=1e3)
parser.add_argument("--D",
                    help="regularization hyper-parameter",
                    type=float, default=1e2)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)

# ====================================================================


def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:12}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    Cfg.C.set_value(args.C)
    Cfg.D.set_value(args.D)
    Cfg.batch_size = args.batch_size
    Cfg.compile_lwsvm = True
    Cfg.softmax_loss = False

    # default value for basefile: string basis for all exported file names
    if args.out_name:
        base_file = "{}/{}".format(args.xp_dir, args.out_name)
    else:
        base_file = "{}/{}_lwsvm".format(args.xp_dir, args.in_name)

    # if pickle file already there, consider run already done
    if (os.path.exists("{}_final_weights.p".format(base_file)) and
        os.path.exists("{}_results.p".format(base_file))):
        sys.exit()

    # computation device
    if 'gpu' in args.device:
        theano.sandbox.cuda.use(args.device)

    np.random.seed(args.seed)

    log_file = "{}/log_{}.txt".format(args.xp_dir, args.dataset)
    if args.dataset != 'imagenet':
        weights = "{}/{}_weights.p".format(args.xp_dir, args.in_name)
    else:
        weights = "{}/vgg16.pkl".format(args.xp_dir)

    nnet = NeuralNet(dataset=args.dataset, use_weights=weights)
    nnet.train(solver="svm")

    # log
    nnet.log.save_to_file("{}_results.p".format(base_file))
    nnet.dump_weights("{}_final_weights.p".format(base_file))

    logged = open(log_file, "a")
    logged.write("{}\t{}\tlwsvm: OK\n".format(args.dataset, args.in_name))
    logged.close()


if __name__ == '__main__':
    main()
