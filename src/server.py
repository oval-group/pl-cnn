import argparse
from datasets.imagenet import ImageNet_Server

parser = argparse.ArgumentParser()
parser.add_argument("--set",
                    help="dataset part ('train' or 'val')",
                    type=str, choices=["train", "val"])


def main():

    args = parser.parse_args()
    server = ImageNet_Server(args.which_set)
    server.start()


if __name__ == '__name__':
    main()
