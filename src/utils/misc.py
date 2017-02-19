import sys


def flush_last_line(to_flush=1):

    for _ in range(to_flush):
        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line
        sys.stdout.flush()
