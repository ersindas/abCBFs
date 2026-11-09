"""Run one controller from the planar quadrotor example."""
from pathlib import Path
import argparse
from model import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['adaptive', 'vanilla', 'nominal'], default='adaptive',
                        help='adaptive bCBF, nominal bCBF, or primary controller only')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('data'))
    args = parser.parse_args()

    if args.mode == 'adaptive':
        from adaptive import run
        run(args.output/'adaptive')
    else:
        from baselines import run
        mode = 'nominal' if args.mode == 'vanilla' else 'primary'
        run(mode, -.37, args.output)


if __name__ == '__main__':
    main()
