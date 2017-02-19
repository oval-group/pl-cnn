import argparse

from utils.visualization.main import paper_plot, xp_plot

parser = argparse.ArgumentParser()
parser.add_argument("--xp_file",
                    help="file for experiment",
                    type=str, default="")
parser.add_argument("--xp_dir",
                    help="directory for experiments",
                    type=str, default="")
parser.add_argument("--export_pdf",
                    help="filename to export the pdf to",
                    type=str, default="")
parser.add_argument('--show', dest='show', action='store_true')
parser.set_defaults(show=False)


def main():
    args = parser.parse_args()

    assert args.xp_file or args.xp_dir

    if args.xp_dir:
        paper_plot(args.xp_dir, args.export_pdf, args.show)

    if args.xp_file:
        xp_plot(args.xp_file, args.export_pdf, args.show)


if __name__ == '__main__':
    main()
