import argparse
import theano

from utils.extraction.imagenet import evaluate
from utils.extraction.results_dump import dump_results
from utils.extraction.latex_table import generate_table

parser = argparse.ArgumentParser()
parser.add_argument("--xp_dir", help="directory for set of experiments",
                    type=str)
parser.add_argument("--weights_file", help="weights file (required for ImageNet)",
                    type=str)
parser.add_argument("--device",
                    help="Computation device to use for experiment",
                    type=str, default="cpu")
parser.add_argument('--imagenet', dest='imagenet', action='store_true')
parser.set_defaults(imagenet=False)


def main():

    args = parser.parse_args()

    # computation device
    if 'gpu' in args.device:
        theano.sandbox.cuda.use(args.device)

    results_name = "{}/results.p".format(args.xp_dir)
    table_name = "{}/table.tex".format(args.xp_dir)

    if args.imagenet:
        evaluate(args.weights_files)
    else:
        dump_results(args.xp_dir, results_name)
        generate_table(results_name, table_name)


if __name__ == '__main__':
    main()
